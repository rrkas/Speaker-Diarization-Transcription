http://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz

gcc -o sph2pipe sph2pipe.c shorten_x.c file_headers.c -lm

./sph2pipe -f rif file.sph file.wav

