@echo off
cd /d "%~dp0"
if not exist "query_tenant_go.exe" (
    echo Building...
    go build -o query_tenant_go.exe .
)
query_tenant_go.exe
