#!/bin/bash

# 检查是否提供了文件名参数
if [ $# -eq 0 ]; then
    echo "错误: 请指定要下载的文件名"
    echo "用法: $0 <文件名>"
    echo "示例: $0 checkpoints_math500_num_generation8_block32_t0.6_lr5e-6_only_rollout_rank.log"
    exit 1
fi

# 获取文件名参数
FILENAME="$1"

# 远程服务器配置
REMOTE_HOST="mc03002@aisurrey-submit01.surrey.ac.uk"
REMOTE_PATH=""
LOCAL_PATH="logs"

# 创建本地目录
mkdir -p "$LOCAL_PATH"

# 远程文件路径
if [ -n "$REMOTE_PATH" ]; then
    REMOTE_FILE="${REMOTE_PATH}/${FILENAME}"
else
    REMOTE_FILE="${FILENAME}"
fi

echo "正在下载:"
echo "  远程: ${REMOTE_HOST}:${REMOTE_FILE}"
echo "  本地: ${LOCAL_PATH}/${FILENAME}"
echo

# 使用 rsync 下载
# -a: 保留文件属性
# -v: 显示详细信息
# -P: 显示进度，并保留未完成文件
# --partial: 中断后保留已下载部分
# --append-verify: 再次运行时从断点继续，并校验已有内容
rsync -avP --partial --append-verify \
    "${REMOTE_HOST}:${REMOTE_FILE}" \
    "${LOCAL_PATH}/"

# 检查结果
if [ $? -eq 0 ]; then
    echo
    echo "✓ 下载成功！"
    echo "文件位置: ${LOCAL_PATH}/${FILENAME}"
    echo
    ls -lh "${LOCAL_PATH}/${FILENAME}"
else
    echo
    echo "✗ 下载中断或失败。"
    echo "不用删除本地未完成文件，直接重新运行同一个命令即可继续下载。"
    exit 1
fi