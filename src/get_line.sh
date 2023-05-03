fn=$1

fshift() { local v n=$'\n';read -r v < <(
    sed -e $'1{w/dev/stdout\n;d}' -i~ "$1")
    printf ${2+-v} $2 "%s${n[${2+2}]}" "$v"
}

fpush() { echo "$2" >> "$1"; }

fshift $fn line
fpush $fn "$line"

echo "$line"
