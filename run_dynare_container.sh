#!/bin/bash

docker run -d --name dynare -p 8888:8888 --shm-size=512M \
    -v $(pwd)/Smets_Wouters_2007:/home/matlab/Documents/Smets_Wouters_2007 \
    dynare/dynare:latest -browser
