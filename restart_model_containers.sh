docker stop prophet arima transformer
docker rm prophet arima transformer

docker run --name prophet -dit -p 7771:8080 prophet-image
docker run --name arima -dit -p 7772:8080 arima-image
docker run --name transformer -dit -p 7773:8080 transformer-image

docker image prune -f