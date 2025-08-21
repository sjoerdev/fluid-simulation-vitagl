#!/bin/bash

VPK=simulation.vpk
IP=192.168.1.162

cmake .

cmake --build .

curl --ftp-method nocwd -T $VPK ftp://$IP:1337/ux0:/transfers/

git clean -Xfd