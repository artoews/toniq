if [ "$1" == "-h" ]; then
  echo "Usage: 'bash make_figures save_dir'"
  exit 0
fi

python scripts/figure1.py $1 -c1 config/mar4-fse125.yml -c2 config/mar4-msl125.yml
python scripts/figure2.py $1 --out out/apr4/mar4-fse125
python scripts/figure4.py $1 -c config/mar4-fse125.yml
# python scripts/figure5.py $1 # slow
python scripts/figure6.py $1 --out1 out/apr4/mar4-fse125 --out2 out/apr4/mar4-msl125
python scripts/figure7.py $1 --out out/apr4/mar4-fse125
python scripts/figure8.py $1 
# python scripts/figure9.py $1 -c config/mar4-fse125.yml -l # very slow without load argument (-l)
python scripts/figure10.py $1 --out1 out/apr4/mar4-msl125 --out2 out/apr4/mar4-msl63