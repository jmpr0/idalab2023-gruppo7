msg=$1

if [ -z "$msg" ]; then
	read -p "Please, set the commit message ['minor']: " msg
	if [ -z "$msg" ]; then
		echo "Using default commit message 'minor'."
		msg="minor"
	fi
fi

git add -u
commit_out=$(git commit -m $msg 2>&1)
echo $commit_out
git pull
git push
commit_hash=$(echo $commit_out | awk -F ' ' '{print $2}' | sed 's|]||g')
git checkout fsl_dev
git cherry-pick $commit_hash
git pull
git push
git checkout main
