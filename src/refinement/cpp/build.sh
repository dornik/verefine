if [ -d build ]; then
    echo "removing old build dir..."
    rm -r build
fi

echo "cmake..."
mkdir build
cd build
cmake ..

echo "make..."
make -j$(getconf _NPROCESSORS_ONLN)
cd ..
echo "done."