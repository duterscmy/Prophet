# #!/bin/bash

# if [ $# -lt 1 ]; then
#   echo "Usage: $0 <log_file>"
#   exit 1
# fi

# log_file="$1"

# if [ ! -f "$log_file" ]; then
#   echo "Error: file not found: $log_file"
#   exit 1
# fi

# grep -oE 'Model forward calls:[[:space:]]*[0-9]+' "$log_file" \
#   | awk -F':' '
#     {
#       gsub(/[[:space:]]/, "", $2)
#       sum += $2
#       count += 1
#     }
#     END {
#       if (count == 0) {
#         print "No NFE lines found."
#         exit 1
#       }
#       printf("count=%d\navg_nfe=%.6f\n", count, sum / count)
#     }
#   '

#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <log_file>"
  exit 1
fi

log_file="$1"

if [ ! -f "$log_file" ]; then
  echo "Error: file not found: $log_file"
  exit 1
fi


grep -oE 'Model forward calls:[[:space:]]*[0-9]+' "$log_file" \
  | awk '
    {
      # extract number after colon
      split($0, a, ":")
      gsub(/[[:space:]]/, "", a[2])
      sum += a[2]
      count += 1
    }
    END {
      if (count == 0) {
        print "No NFE lines found."
        exit 1
      }
      printf("count=%d\navg_nfe=%.6f\n", count, sum / count)
    }
  '