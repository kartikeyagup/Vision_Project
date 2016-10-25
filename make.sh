if [ ! -d build ]; then
  echo "Creating build"
  mkdir build
fi

cd build
cmake ..
rm -rf obsremove

make -j4
if [ -f obsremove ]; then 
  cd ..
  build/obsremove -dirname=init_data/ -num_images=5 -out_dir=out_data/
fi
