


1.Install the latest JRE and get GraphHopper Server as zip from <a href=https://drive.google.com/drive/u/0/folders/1ah7pfjnnJkYiwHUqGLgznPU4u0u5dFme>Graphhopper API</a>. Unzip it.

2.Copy this OSM file into the SAME unzipped directory: <a href=https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf >new-york-latest.osm.pbf</a>

3.Start GraphHopper Maps via: java -jar graphhopper-web-0.10.3-with-dep.jar jetty.resourcebase=webapp config=config-example.properties datareader.file=new-york-latest.osm.pbf.

4.Test to see if its running after you see 'Started server at HTTP 8989' by going to http://localhost:8989/ and you should see a map of New York.

5.Keep this running when executing our program because this is the API

6. Copy the distance files from <a href=https://drive.google.com/drive/u/0/folders/1ah7pfjnnJkYiwHUqGLgznPU4u0u5dFme>Graphhopper API</a>, and store those in folder named Distance.

7. Store the axi data files in the folder named Taxi_Data

8. Run fnal.py
