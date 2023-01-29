#!/bin/bash

rsync -Parvu -e "ssh -p ${1} -i $HOME/.ssh/id_rsa -o StrictHostKeyChecking=no" /home/wombat/data/output/ ${2}@${3}:/home/xlebu/   /

