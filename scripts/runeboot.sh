TITLE=PSVSPHSIM
IP=192.168.1.162

cmake .

cmake --build .

echo destroy | nc $IP 1338

sleep 0.5

curl --ftp-method nocwd -T eboot.bin "ftp://$IP:1337/ux0:/app/$TITLE/"

echo launch $TITLE | nc $IP 1338

git clean -Xfd