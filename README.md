# Splinesurf_libigl

Libigl porting of the splinesurf project performed by Andrea Gatti and Matteo Martini as final project of the Geometric Modeling course, A.Y. 2021/2022, University of Genova.

## Build and run

To build the project it's sufficient to move into the `build` folder and run

``` bash
cmake ..
make
```

To run the program the generated executable file `splinesurf_libigl` can be called. This will load a default Stanford bunny mesh from the `data` folder. If you want to use another mesh (`.obj` files only) you can include its path to the program call like

```bash
./splinesurf_libigl ../data/another_shape.obj
```

## Original project

In the `original` folder you can find the splinesurf source code provided by the authors to us for our project. We fixed some issues and now it compiles using `clang` with just a few warnings using a Linux environment.
