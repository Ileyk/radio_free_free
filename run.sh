rff-flow wind wind.rff --resolution 64 64 64  --inner-radius 1 --outer-radius 10
rff-occult wind.rff occulted-wind.rff --shape sphere --center 0 0 0 --radius 1
./build/rff-transfer --mode spectrum --input occulted-wind.rff --output wind-spectrum.csv   --n0 1e9 --temperature0 1e5 --length0 70   --frequency-min-hz 1e9 --frequency-max-hz 1e12 --frequency-count 100  --view-theta-deg 30 --view-phi-deg 45
rff-plot wind-spectrum.csv