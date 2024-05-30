if [ "$1" == "-h" ]; then
  echo "Usage: 'bash make_figures save_dir main_bool'"
  exit 0
fi

if [ "$2" == "1" ]; then
  # each call to main.py saves outputs to save_dir/config_file_name
  echo 'Running main.py for FSE protocol...'
  python main.py fse125.yml $1
  echo 'Running main.py for MAVRIC-SL protocol...'
  python main.py msl125.yml $1
  echo 'Running main.py for MAVRIC-SL protocol with reduced bandwidth...'
  python main.py msl63.yml $1
fi
 
echo 'Making figure 1...'
python figure1.py $1 -c1 fse125.yml -c2 msl125.yml

echo 'Making figure 2...'
python figure2.py $1 "$1/fse125" -c fse125.yml

echo 'Making figure 4...'
python figure4.py $1 -c fse125.yml

# slow!
echo 'Making figure 5...'
python figure5.py $1

echo 'Making figure 6...'
python figure6.py $1 "$1/fse125" "$1/msl125"

echo 'Making figure 7...'
python figure7.py $1 "$1/fse125" 

echo 'Making figure 8...'
python figure8.py $1

# very slow without load argument (-l)
echo 'Making figure 9...' 
python figure9.py $1 -c fse125.yml # -l

echo 'Making figure 10...'
python figure10.py $1 "$1/msl125" "$1/msl63"

echo 'Done making all figures.'
