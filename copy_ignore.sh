#! /bin/bash

cp -r ./* ~/startup/ray_deployments
cd ~/startup/ray_deployments
git add .
git commit -m "update"
git push origin master

