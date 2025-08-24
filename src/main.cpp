#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

#include <stdio.h>
#include <string.h>

#include <vitasdk.h>
#include <vitaGL.h>
#include <psp2/touch.h>

#include <glm/glm.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <debugfont.hpp>
#include <quickpool.hpp>

bool USE_GLSL = true;

// vitasdk
SceTouchData touch[SCE_TOUCH_PORT_MAX_NUM];
SceCtrlData controller_data;
uint32_t buttons_last = 0;

// other
uint32_t* vitagl_display_framebuf;
int MAX_NEIGHBORS = 10;
quickpool::ThreadPool pool;

std::vector<std::vector<int>> neighbor_buffer;
std::vector<float> position_buffer;
std::vector<float> pressure_buffer;

// opengl
GLuint vao;
GLuint position_vbo;
GLuint pressure_vbo;
GLuint program;
glm::mat4 projection;

// solver parameters
float GRAVITY = -10;
float REST_DENSITY = 300;
float GAS_CONSTANT = 2000;
float KERNEL_RADIUS = 16;
float KERNEL_RADIUS_SQR = KERNEL_RADIUS * KERNEL_RADIUS;
float PARTICLE_MASS = 2.5f;
float VISCOSITY = 200;
float INTIGRATION_TIMESTEP = 0.0007f;

// smoothing kernels and gradients
float POLY6 = 4.f / (glm::pi<float>() * pow(KERNEL_RADIUS, 8.f));
float SPIKY_GRAD = -10.f / (glm::pi<float>() * pow(KERNEL_RADIUS, 5.f));
float VISC_LAP = 40.f / (glm::pi<float>() * pow(KERNEL_RADIUS, 5.f));

// simulation boundary
float BOUNDARY_EPSILON = KERNEL_RADIUS;
float BOUND_DAMPING = -0.5f;

struct Particle
{
    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec2 force;
    float density;
    float pressure;

    Particle(float x, float y)
    {
        position = glm::vec2(x, y);
        velocity = glm::vec2(0.f, 0.f);
        force = glm::vec2(0.f, 0.f);
        density = 0;
        pressure = 0.f;
    }
};

// particles
std::vector<Particle> particles;
int MAX_PARTICLES = 1200;

// projection
int WINDOW_WIDTH = 960;
int WINDOW_HEIGHT = 544;

glm::vec2 GetTouchPosition(int finger)
{
    auto report = touch[SCE_TOUCH_PORT_FRONT].report[finger];
    auto touchpos = glm::vec2(report.x / 2, WINDOW_HEIGHT - report.y / 2);
    return touchpos;
}

bool IsTouchDown()
{
    bool down = touch[SCE_TOUCH_PORT_FRONT].reportNum > 0;
    return down;
}

bool IsButtonPressed(SceCtrlButtons button)
{
    bool thisframe = controller_data.buttons & button;
    bool lastframe = buttons_last & button;
    bool result = thisframe && !lastframe;
    return result;
}

bool IsButtonDown(SceCtrlButtons button)
{
    return controller_data.buttons & button;
}

float RandomValue()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(gen);
}

// grid for spatial hashing
float CELL_SIZE = KERNEL_RADIUS;
int GRID_WIDTH = int(WINDOW_WIDTH / CELL_SIZE) + 1;
int GRID_HEIGHT = int(WINDOW_HEIGHT / CELL_SIZE) + 1;
std::vector<std::vector<int>> grid;

inline int GetCellIndex(int x, int y)
{
    return y * GRID_WIDTH + x;
}

inline glm::ivec2 GetCell(const glm::vec2& pos)
{
    return glm::ivec2(int(pos.x / CELL_SIZE), int(pos.y / CELL_SIZE));
}

void BuildGrid()
{
    grid.clear();
    grid.resize(GRID_WIDTH * GRID_HEIGHT);
    for (int i = 0; i < particles.size(); ++i)
    {
        glm::ivec2 cell = GetCell(particles[i].position);
        cell.x = glm::clamp(cell.x, 0, GRID_WIDTH - 1);
        cell.y = glm::clamp(cell.y, 0, GRID_HEIGHT - 1);
        grid[GetCellIndex(cell.x, cell.y)].push_back(i);
    }
}

inline std::vector<int>& GetNearNeighborsMTBF(Particle& particle, int index)
{
    auto& neighbors = neighbor_buffer[index];
    neighbors.clear();
    
    glm::ivec2 particle_cell = GetCell(particle.position);

    int range = int(ceil(KERNEL_RADIUS / CELL_SIZE));

    for (int offset_x = -range; offset_x <= range; ++offset_x)
    {
        for (int offset_y = -range; offset_y <= range; ++offset_y)
        {
            // calculate cell index
            int cell_x = glm::clamp(particle_cell.x + offset_x, 0, GRID_WIDTH - 1);
            int cell_y = glm::clamp(particle_cell.y + offset_y, 0, GRID_HEIGHT - 1);
            int cell_index = GetCellIndex(cell_x, cell_y);
            
            // for each particle in cell
            for (int j : grid[cell_index])
            {
                // calculate if effected by kernel
                glm::vec2 diff = particles[j].position - particle.position;
                if (glm::dot(diff, diff) < KERNEL_RADIUS_SQR) neighbors.push_back(j);
            }
        }
    }

    return neighbors;
}

void SpawnParticles()
{
    float radius = 60;
    glm::vec2 center = glm::vec2(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);
    float spacing = KERNEL_RADIUS;

    for (float y = center.y - radius; y <= center.y + radius; y += spacing)
    {
        for (float x = center.x - radius; x <= center.x + radius; x += spacing)
        {
            glm::vec2 offset = glm::vec2(RandomValue() - 0.5f, RandomValue() - 0.5f);
            glm::vec2 position = glm::vec2(x, y);
            bool inside = distance(center, position) <= radius;
            if (inside && particles.size() < MAX_PARTICLES) particles.push_back(Particle(x + offset.x, y + offset.y));
        }
    }
}

void ResetParticles()
{
    particles.clear();
    particles.shrink_to_fit();
}

void ComputeDensityPressureMTBF()
{
    BuildGrid();

    pool.parallel_for(0, particles.size(), [](int i)
    {
        Particle& particle_a = particles[i];
        particle_a.density = 0.f;

        auto& neighbors = GetNearNeighborsMTBF(particle_a, i);
        for (int j : neighbors)
        {
            Particle& particle_b = particles[j];
            glm::vec2 rij = particle_b.position - particle_a.position;
            float r2 = glm::dot(rij, rij);
            particle_a.density += PARTICLE_MASS * POLY6 * pow(KERNEL_RADIUS_SQR - r2, 3.f);
        }

        particle_a.pressure = GAS_CONSTANT * (particle_a.density - REST_DENSITY);
    });
}

void ComputeForcesMTBF()
{
    pool.parallel_for(0, particles.size(), [](int i)
    {
        Particle& particle_a = particles[i];
        glm::vec2 pressure_force(0.f);
        glm::vec2 viscosity_force(0.f);

        auto& neighbors = neighbor_buffer[i];
        for (int j : neighbors)
        {
            if (i == j) continue;

            Particle& particle_b = particles[j];
            glm::vec2 diff = particle_b.position - particle_a.position;
            float dist = glm::length(diff);

            if (dist < 1e-6f)
            {
                diff = glm::vec2((RandomValue() - 0.5f) * 0.0001f, (RandomValue() - 0.5f) * 0.0001f);
                dist = glm::length(diff);
            }

            if (dist < KERNEL_RADIUS)
            {
                pressure_force += -normalize(diff) * PARTICLE_MASS * (particle_a.pressure + particle_b.pressure) / (2.f * particle_b.density) * SPIKY_GRAD * std::pow(KERNEL_RADIUS - dist, 3.f);
                viscosity_force += VISCOSITY * PARTICLE_MASS * (particle_b.velocity - particle_a.velocity) / particle_b.density * VISC_LAP * (KERNEL_RADIUS - dist);
            }
        }

        glm::vec2 finger_pos = GetTouchPosition(0);
        glm::vec2 finger_dir = glm::normalize(finger_pos - particle_a.position);
        float finger_dist = glm::distance(finger_pos, particle_a.position);
        bool finger_pressing = IsTouchDown();
        glm::vec2 finger_force = (finger_pressing && finger_dist < 10000) ? finger_dir * PARTICLE_MASS / particle_a.density * 20.f : glm::vec2(0.f);

        glm::vec2 gravity_force = glm::vec2(0.f, GRAVITY) * PARTICLE_MASS / particle_a.density;

        particle_a.force = pressure_force + viscosity_force + gravity_force + finger_force;
    });
}

void IntegrateMTBF()
{
    pool.parallel_for(0, particles.size(), [](int i)
    {
        Particle& particle = particles[i];

        // forward Euler integration
        particle.velocity += INTIGRATION_TIMESTEP * particle.force / particle.density;
        particle.position += INTIGRATION_TIMESTEP * particle.velocity;

        // enforce boundary conditions
        if (particle.position.x - BOUNDARY_EPSILON < 0.f)
        {
            particle.velocity.x *= BOUND_DAMPING;
            particle.position.x = BOUNDARY_EPSILON;
        }
        if (particle.position.x + BOUNDARY_EPSILON > WINDOW_WIDTH)
        {
            particle.velocity.x *= BOUND_DAMPING;
            particle.position.x = WINDOW_WIDTH - BOUNDARY_EPSILON;
        }
        if (particle.position.y - BOUNDARY_EPSILON < 0.f)
        {
            particle.velocity.y *= BOUND_DAMPING;
            particle.position.y = BOUNDARY_EPSILON;
        }
        if (particle.position.y + BOUNDARY_EPSILON > WINDOW_HEIGHT)
        {
            particle.velocity.y *= BOUND_DAMPING;
            particle.position.y = WINDOW_HEIGHT - BOUNDARY_EPSILON;
        }
    });
}

void UpdateMTBF()
{
    ComputeDensityPressureMTBF();
    ComputeForcesMTBF();
    IntegrateMTBF();
}

void RenderMTBF()
{
    glClear(GL_COLOR_BUFFER_BIT);

    if (particles.empty()) return;

    glUseProgram(program);
    glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1, !USE_GLSL, value_ptr(projection));

    // pressure
    float pressureOffset = GAS_CONSTANT * -REST_DENSITY;
    glUniform1f(glGetUniformLocation(program, "minPressure"), pressureOffset + 0);
    glUniform1f(glGetUniformLocation(program, "maxPressure"), pressureOffset + 100);
    glUniform1f(glGetUniformLocation(program, "kernelSize"), KERNEL_RADIUS);

    // prepare buffers
    for (int i = 0; i < particles.size(); i++)
    {
        position_buffer[i * 2] = particles[i].position.x;
        position_buffer[i * 2 + 1] = particles[i].position.y;
        pressure_buffer[i] = particles[i].pressure;
    }

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, position_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particles.size() * 2 * sizeof(float), position_buffer.data());

    glBindBuffer(GL_ARRAY_BUFFER, pressure_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particles.size() * sizeof(float), pressure_buffer.data());

    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particles.size()));

    glBindVertexArray(0);
}

void vgl_draw_character(int character, int x, int y, int scale)
{
    for (int yy = 0; yy < FONT_HEIGHT; yy++)
    {
        for (int ys = 0; ys < scale; ys++)
        {
            int yDisplacement = (y + yy * scale + ys) * WINDOW_WIDTH;
            uint32_t* screenPos = vitagl_display_framebuf + x + yDisplacement;
            uint8_t charPos = font[character * FONT_HEIGHT + yy];

            for (int xx = 0; xx < FONT_WIDTH; xx++)
            {
                // hardcoded bit index calculation
                int bitIndex = 7 - xx;
                uint32_t color = ((charPos >> bitIndex) & 1) ? 0xFFFFFFFF : 0x00000000;
                for (int xs = 0; xs < scale; xs++) screenPos[xs] = color;
                screenPos += scale;
            }
        }
    }
}

void vgl_draw_string(int x, int y, const char *str, int scale)
{
    for (size_t i = 0; i < strlen(str); i++)
    {
        vgl_draw_character(str[i], x + i * FONT_WIDTH * scale, y, scale);
    }
}

void vgl_draw_string_anchored(int cx, int cy, const char *str, int scale, int anchor)
{
    int textWidth = strlen(str) * FONT_WIDTH * scale;
    int y = cy * FONT_HEIGHT * scale;
    int x = cx * FONT_WIDTH * scale;
    
    if (anchor == 0) x = x; // top left
    else if (anchor == 1) x = (WINDOW_WIDTH - textWidth) / 2 + x; // top center
    else if (anchor == 2) x = WINDOW_WIDTH - textWidth - x; // top right

    vgl_draw_string(x, y, str, scale);
}

void vitagl_display_callback(void *framebuf)
{
    vitagl_display_framebuf = (uint32_t*)framebuf;
    
    std::string first = "particles: " + std::to_string(particles.size()) + " / " + std::to_string(MAX_PARTICLES);
    vgl_draw_string_anchored(0, 0, first.c_str(), 2, 1);

    std::string useglsl = USE_GLSL ? "true" : "false";
    std::string second = "using glsl: " + useglsl;
    vgl_draw_string_anchored(0, 1, second.c_str(), 2, 1);

    std::string third = "threads: " + std::to_string(pool.get_active_threads());
    vgl_draw_string_anchored(0, 2, third.c_str(), 2, 1);
}

GLuint CompileShader(std::string source, GLenum type)
{
    GLuint shader = glCreateShader(type);
    const char* src = source.c_str();

    if (USE_GLSL) glShaderSource(shader, 1, &src, nullptr);
    else vglCgShaderSource(shader, 1, &src, nullptr);

    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string infoLog(length, ' ');
        glGetShaderInfoLog(shader, length, nullptr, &infoLog[0]);
        glDeleteShader(shader);
        std::string log = infoLog;
    }

    return shader;
}

GLuint CompileProgram(std::string vertCode, std::string fragCode)
{
    GLuint vertex = CompileShader(vertCode, GL_VERTEX_SHADER);
    GLuint fragment = CompileShader(fragCode, GL_FRAGMENT_SHADER);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint length;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        std::string infoLog(length, ' ');
        glGetProgramInfoLog(program, length, nullptr, &infoLog[0]);
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        std::string log = infoLog;
    }

    glDeleteShader(vertex);
    glDeleteShader(fragment);

    return program;
}

std::string VertShaderCG()
{
    return R"(
        void main
        (
            float2 aPos : ATTR0,
            float  aPressure : ATTR1,

            uniform float4x4 projection,
            uniform float minPressure,
            uniform float maxPressure,
            uniform float kernelSize,

            out float4 gl_Position : POSITION,
            out float3 VertColor : TEXCOORD0,
            out float  PointSize : PSIZE
        )
        {
            gl_Position = mul(projection, float4(aPos, 0.0, 1.0));

            float clamped_pressure = saturate((aPressure - minPressure) / (maxPressure - minPressure));
            VertColor = lerp(float3(0.0, 0.4, 1.0), float3(1.0, 1.0, 1.0), clamped_pressure);

            PointSize = kernelSize / 2.0;
        }
    )";
}

std::string FragShaderCG()
{
    return R"(
        float4 main
        (
            float3 VertColor : TEXCOORD0,
            float2 gl_PointCoord : SPRITECOORD
        ) : COLOR
        {
            float2 coord = gl_PointCoord - float2(0.5, 0.5);
            float dist = length(coord);
            if (dist > 0.5) discard;

            return float4(VertColor, 1.0);
        }
    )";
}

std::string VertShaderGLSL()
{
    std::string temp = 
    R"(
        #version 120

        attribute vec2 aPos;
        attribute float aPressure;

        varying vec3 VertColor;

        uniform mat4 projection;
        uniform float minPressure;
        uniform float maxPressure;
        uniform float kernelSize;

        void main()
        {
            gl_Position = projection * vec4(aPos, 0.0, 1.0);
            gl_PointSize = kernelSize / 2.0;

            float clamped_pressure = clamp((aPressure - minPressure) / (maxPressure - minPressure), 0.0, 1.0);
            VertColor = mix(vec3(0.0, 0.4, 1.0), vec3(1.0, 1.0, 1.0), clamped_pressure);
        }
    )";
    return temp;
}

std::string FragShaderGLSL()
{
    std::string temp = 
    R"(
        #version 120

        varying vec3 VertColor;

        void main()
        {
            // discard if outside radius
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);
            if (dist > 0.5) discard;

            gl_FragColor = vec4(VertColor, 1.0);
        }
    )";
    return temp;
}

void SetupBuffers()
{
    // bind and gen vao
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // position buffer
    glGenBuffers(1, &position_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, position_vbo);
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * 2 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    // pressure buffer
    glGenBuffers(1, &pressure_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, pressure_vbo);
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);

    // unbind vao
    glBindVertexArray(0);
}

int main()
{
    // init vitagl
    if (USE_GLSL) vglSetSemanticBindingMode(VGL_MODE_POSTPONED);
    vglInit(0x800000);
    vglSetDisplayCallback(vitagl_display_callback);
    
    // init opengl
    glClearColor(0, 0, 0, 1);
    if (USE_GLSL) program = CompileProgram(VertShaderGLSL(), FragShaderGLSL());
    else program = CompileProgram(VertShaderCG(), FragShaderCG());
    projection = glm::ortho(0.0f, float(WINDOW_WIDTH), 0.0f, float(WINDOW_HEIGHT), -1.0f, 1.0f);
    SetupBuffers();
    
    // init input
    sceCtrlSetSamplingMode(SCE_CTRL_MODE_ANALOG);
    sceTouchSetSamplingState(SCE_TOUCH_PORT_FRONT, SCE_TOUCH_SAMPLING_STATE_START);
    sceTouchEnableTouchForce(SCE_TOUCH_PORT_FRONT);

    // allocate particle buffers
    neighbor_buffer.resize(MAX_PARTICLES);
    position_buffer.resize(MAX_PARTICLES * 2);
    pressure_buffer.resize(MAX_PARTICLES);

    // init simulation
    SpawnParticles();

    while (true)
    {
        sceCtrlPeekBufferPositive(0, &controller_data, 1);
        sceTouchPeek(SCE_TOUCH_PORT_FRONT, &touch[SCE_TOUCH_PORT_FRONT], 1);

        if (IsButtonPressed(SCE_CTRL_CROSS)) SpawnParticles();
        if (IsButtonPressed(SCE_CTRL_CIRCLE)) ResetParticles();

        UpdateMTBF();
        RenderMTBF();

        vglSwapBuffers(GL_FALSE);
        buttons_last = controller_data.buttons;
    }

    vglEnd();
}