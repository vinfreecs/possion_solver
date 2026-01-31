set terminal png size 1024,768 enhanced font ,12
set output 'init.png'
set datafile separator whitespace

set grid
set hidden3d
splot 'init.dat' matrix using 1:2:3 with lines
