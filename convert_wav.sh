echo 'CONVERTING NOISY WAVS TO 16K...'
mkdir -p preprocessed_noisy_testset
pushd noisy_testset
ls *.wav | while read name; do
    sox $name -r 16k ../preprocessed_noisy_testset/$name
done
popd