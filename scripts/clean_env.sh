#!/bin/bash

# 清理 conda 环境的脚本
# 此脚本会删除所有非 base 的 conda 环境

# 获取所有环境列表
ENVS=$(conda env list | grep -v "^#" | grep -v "^$" | grep -v "base" | awk '{print $1}')

echo "找到以下 conda 环境："
echo "$ENVS"
echo ""

# 确认是否继续
read -p "是否删除所有非 base 环境？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 1
fi

# 删除每个环境
for env in $ENVS; do
    echo "正在删除环境: $env"
    conda env remove -n $env
done

echo "环境清理完成！"
echo "当前环境列表："
conda env list 