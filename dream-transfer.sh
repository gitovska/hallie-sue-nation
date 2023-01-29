#!/bin/bash

rsync -Parvu -e "ssh -p ${1}" /home/wombat/data/output/ ${2}@${3}:/home/xlebu/docker-nginx/html/

