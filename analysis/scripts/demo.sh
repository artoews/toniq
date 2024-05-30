echo 'Running main.py for FSE protocol...'
python main.py fse125.yml demo

echo 'Making figure 2...'
python figure2.py demo "demo/fse125" -c fse125.yml

echo 'Demo complete. Check images in demo folder.'