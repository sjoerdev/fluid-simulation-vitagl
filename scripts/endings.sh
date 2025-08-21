for script in scripts/*.sh; do
    sed -i 's/\r$//' "$script"
done