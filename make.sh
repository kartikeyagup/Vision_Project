if [ ! -d build ]; then
  echo "Creating build"
  mkdir build
fi

if [ ! -d out_data ]; then 
  mkdir out_data
fi

cd build
cmake ..
rm -rf obsremove
rm -rd cere_try

make -j4
if [ -f obsremove ]; then 
  cd ..
  # build/obsremove -dirname=init_data2/ -num_images=5 -out_dir=out_data/
  build/ceres_try
fi
