@echo off
REM 安装requirements.txt中的依赖
pip install -r requirements.txt

REM 运行streamlit应用
streamlit run streamlit.py

pause