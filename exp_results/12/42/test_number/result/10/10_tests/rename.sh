OLD="our_solution"
NEW="10_samples"

for file in *"$OLD"*; do
  # 디렉토리가 아닌 '파일'인 경우에만 실행
  if [ -f "$file" ]; then
    # ${file//$OLD/$NEW} : 파일명(file)에서 OLD 문자열 전체를 NEW로 바꿈
    mv "$file" "${file//$OLD/$NEW}"
  fi
done