# ParallelScan.cu

The goal of this project is to implement a kernel to perform an inclusive parallel scan on a 1D list (with randomly
generated integers) using both the work-efficient and work-inefficient algorithms in CUDA

## Platform
Project created on Windows X64

## IDE Used
Visual Studio 2022

## Build the code 
-> Extract the zip folder on a Windows machine 
-> Build using Visual Studio 2022 [use the 2022 version]
-> Now open the terminal and traverse to the Debug directory.

## Usage
Sample command to create 
Sample userinput size: 9999
******************************************
```bash
cd .\x64\
cd .\Debug\

# Now in the Debug directory
 .\ParallelScan.exe scan-work-inefficient -i 9999
 or
  .\ParallelScan.exe scan-work-efficient -i 9999
```
******************************************
## NOTE
Please make sure that the project settings are also imported, if not then follow this
******************************************
--> Open Solution Explorer 
--> Navigate to projectOne properties page 
--> Under VC++ Directories 
--> Find External Included Directories 
--> Add the Project_three\Common folder to the list(Already present in the Zipped folder) 
--> Necessary to include 
    #include <helper_functions.h>
    #include <helper_cuda.h>
