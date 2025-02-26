while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S')" >> container_stats.log
    
    # Log Docker container stats
    docker stats --no-stream >> container_stats.log
    
    # Log GPU memory usage
    echo "GPU Memory Usage:" >> container_stats.log
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F ',' '{printf "GPU: %s/%s MiB (%s%%)\n", $1, $2, $3}' >> container_stats.log
    
    echo "-----------------------------------" >> container_stats.log
    
    sleep 0.5
done
