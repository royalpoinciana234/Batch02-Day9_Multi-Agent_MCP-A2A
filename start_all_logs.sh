#!/bin/bash
set -e

mkdir -p stages/stage_5_logs

echo "Starting Registry service on port 10000..."
python3 -m registry > stages/stage_5_logs/registry.log 2>&1 &
REGISTRY_PID=$!
sleep 2

echo "Starting Tax Agent on port 10102..."
python3 -m tax_agent > stages/stage_5_logs/tax_agent.log 2>&1 &
TAX_PID=$!

echo "Starting Compliance Agent on port 10103..."
python3 -m compliance_agent > stages/stage_5_logs/compliance_agent.log 2>&1 &
COMPLIANCE_PID=$!
sleep 3

echo "Starting Law Agent on port 10101..."
python3 -m law_agent > stages/stage_5_logs/law_agent.log 2>&1 &
LAW_PID=$!
sleep 3

echo "Starting Customer Agent on port 10100..."
python3 -m customer_agent > stages/stage_5_logs/customer_agent.log 2>&1 &
CUSTOMER_PID=$!

echo "All services started. Logs are redirected to stages/stage_5_logs/"
wait $REGISTRY_PID $TAX_PID $COMPLIANCE_PID $LAW_PID $CUSTOMER_PID
