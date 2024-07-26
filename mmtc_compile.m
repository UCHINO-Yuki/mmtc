function mmtc_compile

disp('now: compiling mmtc.cu ...')
try
    mexcuda mmtc.cu ...
        -R2018a -lcuda -lcudart -lcublas ...
        -silent ...
        CXXFLAGS='$CXXFLAGS -O3' ...
        COMPFLAGS='$COMPFLAGS -O3'
catch
    error('error');
    disp('try this command: setenv("NVCC_APPEND_FLAGS", ''-allow-unsupported-compiler'');');
end

disp('done!');
