packages_path=$(python3 -m site | grep "packages',")
packages_path=${packages_path:5:-2}
file_path="sklearn/metrics/cluster/_unsupervised.py"
full_path="$packages_path/$file_path"

if [ -f "$full_path.bk" ]; then
    mv $full_path.bk $full_path
fi

cp $full_path $full_path.bk

sed "s|def davies_bouldin_score(X, labels):|def davies_bouldin_score(X, labels, average='macro'):|g" $full_path.bk > $full_path
sed -i "s|    return np.mean(scores)|    if average is None:\n        return scores\n    return np.mean(scores)|g" $full_path

ls "$full_path"*
