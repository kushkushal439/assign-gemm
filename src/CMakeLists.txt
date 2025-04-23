add_library(studentlib main_server.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(studentlib PRIVATE -O3 -march=native -funroll-loops -ffast-math -fopenmp -flto -fno-signed-zeros -fno-trapping-math)

target_compile_options(studentlib PRIVATE 
    -finline-functions 
    -finline-limit=1000
    -fpeel-loops
    -fpredictive-commoning
)

target_compile_options(studentlib PRIVATE
    -fno-math-errno
    -ffinite-math-only
    -freciprocal-math
    -fno-signed-zeros
    -fassociative-math
    -malign-data=cacheline
)

# Proper OpenMP support
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(studentlib PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(studentlib PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Important: Also add LTO to link flags
target_link_options(studentlib PRIVATE -flto)

pragma GCC target("avx512f,avx512cd,avx512vl,avx512bw,avx512dq,fma")