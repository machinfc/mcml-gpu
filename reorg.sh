#!/bin/bash
# Reorganize project
mkdir -p src/mcml src/conv include data scripts build

# Move mcml source and headers
mv mcmlmain.c mcmlgo.c mcmlio.c mcmlnr.c src/mcml/
mv mcml.h include/

# Move conv source and headers
mv convconv.c extra.c src/conv/
mv conv.h extra.h include/

# Move test files if any
# test.c is garbage, let's just delete it along with object files
rm -f test.c test *.o mcml test_100K.mco test_1M.mco random out

# Move data files
mv *.mci *.mco *.mco2 *.mco3 *.Frzc data/ 2>/dev/null

# Move paralelo stuff to data/paralelo or scripts?
# notebooks to scripts
mkdir -p scripts/notebooks
find paralelo -name "*.nb" -exec mv {} scripts/notebooks/ \;
mv paralelo data/ 2>/dev/null

# Remove old IDE and Makefile
rm -f Makefile mcmlcode.kdev4

echo "Reorganization complete"
