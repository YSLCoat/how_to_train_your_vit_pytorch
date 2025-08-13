@echo off
setlocal

set MODEL_NAME=lucidrain_vit
set BATCH=256
set DATA_FOLDER=C:\data\imagenet1k\ILSVRC\Data\CLS-LOC

python "C:\easy_train\train.py" ^
  --arch "%MODEL_NAME%" ^
  --batch-size %BATCH% ^
  --data "%DATA_FOLDER%"

set EXITCODE=%ERRORLEVEL%
echo Exit code: %EXITCODE%
pause
exit /b %EXITCODE%
