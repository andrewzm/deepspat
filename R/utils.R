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

logdet <- function (R) {
  diagR <- diag(R)
  return(2 * sum(log(diagR)))
}

tr <- function(A) {
  sum(diag(A))
}

safe_chol <- function(A) {
  A <- A + 10^(-6) * diag(nrow(A))
  chol(A)
}

atBa <- function(a, B) {
  t(a) %*% (B %*% a)
}

ABinvAt <- function(A, cholB) {
  tcrossprod(A %*% solve(cholB))
}

AtBA_p_C <- function(A, cholB, C) {
  crossprod(cholB %*% A) + C
}

entropy <- function(s) {
  d <- ncol(s)
  0.5 * sum(colSums(log(s)))
}

get_depvars <- function(f) {
  . <- NULL
  stopifnot(is(f, "formula"))
  if(!attr(terms(f), "response")) {
    depvars <- NULL
  } else {
    gr <- grepl("cbind", as.character(f))
    idx <- which(gr)
    if(length(idx) > 0) {
      terms(f)
      depvars <- ((attr(terms(f), "variables") %>%
                     as.character())[2] %>%
                    strsplit(","))[[1]] %>%
        gsub("cbind\\(|\\)|\\s", "", .)
    } else  {
      depvars <- all.vars(f)[[1]]
    }
  }
  depvars
}

get_depvars_multivar <- function(f) {
  . <- NULL
  stopifnot(is(f, "formula"))
  if(!attr(terms(f), "response")) {
    depvars <- NULL
  } else {
    gr <- grepl("cbind", as.character(f))
    idx <- which(gr)
    if(length(idx) > 0) {
      terms(f)
      depvars <- ((attr(terms(f), "variables") %>%
                     as.character())[2] %>%
                    strsplit(","))[[1]] %>%
        gsub("cbind\\(|\\)|\\s", "", .)
    } else  {
      depvars <- c(all.vars(f)[[1]], all.vars(f)[[2]])
    }
  }
  depvars
}

get_depvars_multivar2 <- function(f) {
  . <- NULL
  stopifnot(is(f, "formula"))
  if(!attr(terms(f), "response")) {
    depvars <- NULL
  } else {
    gr <- grepl("cbind", as.character(f))
    idx <- which(gr)
    if(length(idx) > 0) {
      terms(f)
      depvars <- ((attr(terms(f), "variables") %>%
                     as.character())[2] %>%
                    strsplit(","))[[1]] %>%
        gsub("cbind\\(|\\)|\\s", "", .)
    } else  {
      depvars <- c(all.vars(f)[[1]], all.vars(f)[[2]], all.vars(f)[[3]])
    }
  }
  depvars
}



pinvsolve <- function(A, b, reltol = 1e-6) {
  # Compute the SVD of the input matrix A
  A_SVD = svd(A)
  s <- A_SVD$d
  u <- A_SVD$u
  v <- A_SVD$v

  # Invert s, clear entries lower than reltol*s[0].
  atol = max(s) * reltol
  s_mask = s[which(s > atol)]
  s_reciprocal <- 1/s_mask
  s_inv = diag(c(s_reciprocal, rep(0, length(s) - length(s_mask))))

  # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
  v %*% (s_inv %*% (t(u) %*% b))
}


list_to_listtf <- function(l, name, constant = TRUE, dtype = "float32") {
  stopifnot(is.list(l))
  stopifnot(is.character(name))
  stopifnot(is.logical(constant))

  if(constant) tffun <- tf$constant else tffun <- tf$Variable
  lapply(1:length(l), function(i)
    tffun(l[[i]], name = paste0(name, i), dtype = dtype))
}

proc_m.inducing <- function(m.inducing = 10L, nlayers = 1) {
  if(length(m.inducing) == 1)
    m.inducing <- rep(m.inducing, nlayers)
  m.inducing
}

scal_0_5 <- function(s) {
  mins <- min(s)
  maxs <- max(s)
  s <- (s - min(s)) / (maxs - mins) - 0.5
}

scal_0_5_mat <- function(s) {
  mins <- matrix(1, nrow(s), 1) %*% apply(s, 2, min)
  maxs <- matrix(1, nrow(s), 1) %*% apply(s, 2, max)
  s <- (s - mins) / (maxs - mins) - 0.5
}


KL <- function(mu1, S1, mu2, S2) {
  0.5*(sum(diag(solve(S2) %*% S1)) +
        t(mu2 - mu1) %*% solve(S2) %*% (mu2 - mu1) -
        nrow(mu1) +
        determinant(S2)$modulus -
         determinant(S1)$modulus)
}

## Plot warping in ggplot
polygons_from_points <- function(df, every = 3) {
  # df must have s1c and s2c that are integers
  #               s1 and s2
  #               h1 and h2

  s1c <- s2c <- NULL
  df <- df %>%  filter((s1c %% every == 0) & (s2c %% every == 0))

  cells <- list()
  count <- 0
  for(i in 1:nrow(df)) {
    this_centroid <- df[i,]
    d <- filter(df, (s1c - this_centroid$s1c) < (every + 1) & (s1c - this_centroid$s1c)  >= 0 &
                  (s2c - this_centroid$s2c) < (every + 1) & (s2c - this_centroid$s2c >= 0))

    if(nrow(d) == 4)  {
      count <- count + 1
      idx1 <- which(d$s1 == min(d$s1) & d$s2 == min(d$s2))
      idx2 <- which(d$s1 == max(d$s1) & d$s2 == min(d$s2))
      idx3 <- which(d$s1 == max(d$s1) & d$s2 == max(d$s2))
      idx4 <- which(d$s1 == min(d$s1) & d$s2 == max(d$s2))

      this_cell <- data.frame(x = d$h1[c(idx1, idx2, idx3, idx4)],
                              y = d$h2[c(idx1, idx2, idx3, idx4)],
                              s1c = d$s1c[c(idx1, idx2, idx3, idx4)],
                              s2c = d$s2c[c(idx1, idx2, idx3, idx4)],
                              id = count)
      cells[[count]] <- this_cell
    }
  }
  data.table::rbindlist(cells)
}

