SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for rho in 95 99; do
    bash "${SCRIPT_DIR}/dogscats_train.sh" "$rho"
done