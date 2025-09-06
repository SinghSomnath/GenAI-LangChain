@echo off
set varr=%USERNAME%
icacls bastion-key.pem /inheritance:r
icacls bastion-key.pem /grant:r %varr%:(R,W)
icacls bastion-key.pem /remove "BUILTIN\Users"
echo Starting SSH tunnel to Bastion...
ssh -i bastion-key.pem -L 3306:%1:3306 ubuntu@%2
