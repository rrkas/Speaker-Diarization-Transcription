#!/bin/bash

set -x 
set -e

storage_dir=../../../data/
mkdir -p $storage_dir

wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/dev-clean.tar.gz -P $storage_dir
tar -xzf $storage_dir/dev-clean.tar.gz -C $storage_dir

wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/test-clean.tar.gz -P $storage_dir
tar -xzf $storage_dir/test-clean.tar.gz -C $storage_dir

wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-100.tar.gz -P $storage_dir
tar -xzf $storage_dir/train-clean-100.tar.gz -C $storage_dir

wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-360.tar.gz -P $storage_dir
tar -xzf $storage_dir/train-clean-360.tar.gz -C $storage_dir

rm $storage_dir/*.tar.gz


