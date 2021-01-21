# What digit you write?

![](https://img.shields.io/badge/TensorFlow-2.1-orange) ![](https://img.shields.io/badge/Python-3.7-blue)

Handwritten digit recognition application implemented by TensorFlow2 + Keras and Flask.

![](https://cdn.jsdelivr.net/gh/innofang/jotter/source/waht-digit-you-write/screencast.gif)

## How to run?

```shell script
$ git clone --depth 1 https://github.com/InnoFang/what-digit-you-write.git
$ cd what-digit-you-write
$ conda create --name <env> --file requirements.txt
$ conda activate <env>
$ python app.py
```

If the clone is too slow, you can use the following method

```shell script
$ # git clone --depth 1 https://github.com.cnpmjs.org/InnoFang/what-digit-you-write.git 
```

### docker env

see `Dockerfile` for how the environment is built

````
$ git clone --depth 1 https://github.com/InnoFang/what-digit-you-write.git
$ cd what-digit-you-write
$ docker build -t ml-digit .
$ docker run -it -p 5000:5000 ml-digit
````