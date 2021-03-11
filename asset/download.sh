file_id="10ofPKSFnYMhm0Yh8IvdHBxJJhFjG1Qck"
file_name="checkpoint_ssd300.pth.tar"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${file_id}" -o ${file_name}
rm cookie