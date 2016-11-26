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
rm -rf ceres_try

make -j4
if [ -f obsremove ]; then 
  cd ..
  build/obsremove -dirname=init_data2_m/ -num_images=3 -out_dir=out_data/
  # build/ceres_try
fi
