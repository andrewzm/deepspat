## Copyright 2019 Andrew Zammit Mangion
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

tent <- function(x, theta) {
  (abs(x - theta[1]) < theta[2]) *
    ((x < theta[1])*((x - theta[1] + theta[2])/theta[2]) -
       (x > theta[1])*((x - theta[1] - theta[2])/theta[2]))
}

tent_tf <- function(x, theta) {
  leftbit <- tf$cast((x <= theta[1] & x >= theta[1] - theta[2]), "float32")*((x - theta[1] + theta[2])/theta[2])
  rightbit <- -tf$cast((x > theta[1] & x < theta[1] + theta[2]), "float32")*((x - theta[1] - theta[2])/theta[2])
  tf$multiply(x, 0) %>% tf$add(leftbit) %>% tf$add(rightbit)
}

