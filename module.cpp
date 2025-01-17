#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

// Uncomment for ISPC
// #include "module_ispc.h"
// using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    return tensor[
        x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b
    ];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    tensor[
        x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b
    ] = val; 
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //
const bool USE_MIN_NUMERIC = 0;

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    // QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    // The code below is used instead of the uncommented code below it for ISPC
    // part1(Q.data(), K.data(), V.data(), QK_t.data(), O.data(), B, H, N, d);

    // -------- YOUR CODE HERE  -------- //
    for (int b = 0; b < B; b++) {
      for (int h = 0; h < H; h++) {
        
        // QK^t mul
        for (int i = 0; i < N; i++) {
          for (int k = 0; k < N; k++) {
            val = 0.f;
            for (int j = 0; j < d; j++) {
              float q = fourDimRead(Q, b, h, i, j, H, N, d);
              float k_t = fourDimRead(K, b, h, k, j, H, N, d);
              val += q * k_t;
            }
            twoDimWrite(QK_t, i, k, N, val);
          }
        }
        
        // Softmax
        for (int i = 0; i < N; i++) {
          val = 0.f;
          float sample, normed;
          float min_val = 0.f;
          if (USE_MIN_NUMERIC) {
            min_val = std::numeric_limits<float>::max();;
            for (int j = 0; j < N; j++) {
              sample = twoDimRead(QK_t, i, j, N);
              if (sample < min_val) {min_val = sample;}
            }
          }
          for (int j = 0; j < N; j++) {
            val += exp(twoDimRead(QK_t, i, j, N) - min_val);
          }
          for (int j = 0; j < N; j++) {
            normed = exp(twoDimRead(QK_t, i, j, N) - min_val) / val;
            twoDimWrite(QK_t, i, j, N, normed);
          }
        }

        // QK_t V matmul
        for (int i = 0; i < N; i++) {
          for (int k = 0; k < d; k++) {
            val = 0.f;
            for (int j = 0; j < N; j++) {
              float q = twoDimRead(QK_t, i, j, N);
              float v = fourDimRead(V, b, h, j, k, H, N, d);
              val += q * v;
            }
            fourDimWrite(O, b, h, i, k, H, N, d, val);
          }
        }

      }
    }
    // -------- YOUR CODE HERE  -------- //

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //
const int BLOCK = 8;

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // Uncomment for ISPC
    // part2(Q.data(), K.data(), V.data(), QK_t.data(), O.data(), B, H, N, d, BLOCK);

    // -------- YOUR CODE HERE  -------- //
    float val;
    for (int b = 0; b < B; b++) {
      for (int h = 0; h < H; h++) {
        std::fill(QK_t.begin(), QK_t.end(), 0);
        
        // QK^t mul
        for (int ii = 0; ii < N; ii += BLOCK) {
          uint BLOCK_I = min(ii + BLOCK, N);
          for (int kk = 0; kk < N; kk += BLOCK) {
            uint BLOCK_K = min(kk + BLOCK, N);
            for (int jj = 0; jj < d; jj += BLOCK) {
              uint BLOCK_J = min(jj + BLOCK, d);
              for(int i = ii; i < BLOCK_I; i++){
                for(int k = kk; k < BLOCK_K; k++){
                  val = twoDimRead(QK_t, i, k, N);
                  for(int j = jj; j < BLOCK_J; j++){
                    val += fourDimRead(Q, b, h, i, j, H, N, d) * fourDimRead(K, b, h, k, j, H, N, d);
                  }
                  twoDimWrite(QK_t, i, k, N, val);
                }
              }
            }
          }
        }
        
        // cout << QK_t << endl;

        
        // Softmax
        for (int i = 0; i < N; i++) {
          val = 0.f;
          float sample, normed;
          float min_val = 0.f;
          if (USE_MIN_NUMERIC) {
            min_val = std::numeric_limits<float>::max();;
            for (int j = 0; j < N; j++) {
              sample = twoDimRead(QK_t, i, j, N);
              if (sample < min_val) {min_val = sample;}
            }
          }
          for (int j = 0; j < N; j++) {
            val += exp(twoDimRead(QK_t, i, j, N) - min_val);
          }
          for (int j = 0; j < N; j++) {
            normed = exp(twoDimRead(QK_t, i, j, N) - min_val) / val;
            twoDimWrite(QK_t, i, j, N, normed);
          }
        }

        // QK_t V matmul
        for (int ii = 0; ii < N; ii += BLOCK) {
          uint BLOCK_I = min(ii + BLOCK, N);
          for (int kk = 0; kk < d; kk += BLOCK) {
            uint BLOCK_K = min(kk + BLOCK, d);
            for (int jj = 0; jj < N; jj += BLOCK) {
              uint BLOCK_J = min(jj + BLOCK, N);
              for(int i = ii; i < BLOCK_I; i++){
                for(int k = kk; k < BLOCK_K; k++){
                  val = fourDimRead(O, b, h, i, k, H, N, d);
                  for(int j = jj; j < BLOCK_J; j++){
                    val += twoDimRead(QK_t, i, j, N) * fourDimRead(V, b, h, j, k, H, N, d);
                  }
                  fourDimWrite(O, b, h, i, k, H, N, d, val);
                }
              }
            }
          }
        }

      }
    }
    // -------- YOUR CODE HERE  -------- //

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    // Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);

    // The code below is used instead of the uncommented code below it for ISPC
    // #pragma omp parallel for collapse(3)
    // for (int b = 0; b < B; b++){   
    //   for (int h = 0; h < H; h++){
    //     for (int i = 0; i < N ; i++){
    //       at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
    //       std::vector<float> QK_t= formatTensor(ORowTensor);
    //       std::fill(QK_t.begin(), QK_t.end(), 0);
    //
    //       float val;
    //       fusedMatrixMult(Q.data(), K.data(), QK_t.data(), B, H, N, d, b, h, BLOCK, i);
    //       fusedSoftmaxNorm(QK_t.data(), N, i);
    //       fusedPvCalc(QK_t.data(), V.data(), O.data(), B, H, N, d, b, h, BLOCK, i);
    //     }
	//   }
    // }

    // -------- YOUR CODE HERE  -------- //
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++){
      for (int h = 0; h < H; h++){
        for (int i = 0; i < N ; i++){
          at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
          std::vector<float> QK_t= formatTensor(ORowTensor);
          std::fill(QK_t.begin(), QK_t.end(), 0);

          float val;
          
          for (int kk = 0; kk < N; kk += BLOCK) {
            uint BLOCK_K = min(kk + BLOCK, N);
            for (int jj = 0; jj < d; jj += BLOCK) {
              uint BLOCK_J = min(jj + BLOCK, d);
              for(int k = kk; k < BLOCK_K; k++){
                val = QK_t[k];
                for(int j = jj; j < BLOCK_J; j++){
                  val += fourDimRead(Q, b, h, i, j, H, N, d) * fourDimRead(K, b, h, k, j, H, N, d);
                }
                QK_t[k] = val;
              }
            }
          }

          val = 0.f;
          float sample, normed;
          float min_val = 0.f;
          if (USE_MIN_NUMERIC) {
            min_val = std::numeric_limits<float>::max();;
            for (int j = 0; j < N; j++) {
              sample = QK_t[j];
              if (sample < min_val) {min_val = sample;}
            }
          }
          for (int j = 0; j < N; j++) {
            val += exp(QK_t[j] - min_val);
          }
          for (int j = 0; j < N; j++) {
            QK_t[j] = exp(QK_t[j] - min_val) / val;
          }

          for (int kk = 0; kk < d; kk += BLOCK) {
            uint BLOCK_K = min(kk + BLOCK, d);
            for (int jj = 0; jj < N; jj += BLOCK) {
              uint BLOCK_J = min(jj + BLOCK, N);
              for(int k = kk; k < BLOCK_K; k++){
                val = fourDimRead(O, b, h, i, k, H, N, d);
                for(int j = jj; j < BLOCK_J; j++){
                  val += QK_t[j] * fourDimRead(V, b, h, j, k, H, N, d);
                }
                fourDimWrite(O, b, h, i, k, H, N, d, val);
              }
            }
          }


        }
	  }
    }
    // -------- YOUR CODE HERE  -------- //
    
	    
	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // The code below is used instead of the uncommented code below it for ISPC
    // for (int b = 0; b < B; b++) {
    //   for (int h = 0; h < H; h++) {
    //     part4(O.data(), Q.data(), K.data(), V.data(), Sij.data(), Pij.data(), Kj.data(), Vj.data(), 
    //             Qi.data(), Oi.data(), l.data(), PV.data(), li.data(), lij.data(), lnew.data(),
    //             Bc, Br, B, H, N, d, b, h);
    //   }
    // }

    // -------- YOUR CODE HERE  -------- //
    for (int b = 0; b < B; b++) {
      for (int h = 0; h < H; h++) {
        int step = b * (H * N * d) + h * (N * d);
        std::fill(l.begin(), l.end(), 0.f);

        for (int jj = 0; jj < N; jj += Bc) {
          uint BLOCK_J = min(Bc, N - jj);

          for (int j = 0; j < BLOCK_J; j++) {
            for (int k = 0; k < d; k++) {
              Kj[j * d + k] = K[step + (jj + j) * d + k];
              Vj[j * d + k] = V[step + (jj + j) * d + k];
            }
          }

          for (int ii = 0; ii < N; ii += Br) {
            uint BLOCK_I = min(Br, N - ii);

            for (int i = 0; i < BLOCK_I; i++) {
              li[i] = l[ii + i];
              for (int k = 0; k < d; k++) {
                Qi[i * d + k] = Q[step + (ii + i) * d + k];
                Oi[i * d + k] = O[step + (ii + i) * d + k];
              }
            }

            // QiKi^T
            for (int i = 0; i < BLOCK_I; i++) {
              for (int j = 0; j < BLOCK_J; j++) {
                float val = 0.f;
                for (int k = 0; k < d; k++) {
                  val += Qi[i * d + k] * Kj[j * d + k];
                }
                Sij[i * BLOCK_J + j] = val;
              }
            }

            // QiKi^T
            for (int i = 0; i < BLOCK_I; i++) {
              for (int j = 0; j < BLOCK_J; j++) {
                Pij[i * BLOCK_J + j] = exp(Sij[i * BLOCK_J + j]);
              }
            }
          
            // Rowsum
            for (int i = 0; i < BLOCK_I; i++) {
              float val = 0.f;
              for (int j = 0; j < BLOCK_J; j++) {
                val += Pij[i * BLOCK_J + j];
              }
              lij[i] = val;
            }

            for (int i = 0; i < BLOCK_I; i++) {
              lnew[i] = li[i] + lij[i];
            }

            for (int i = 0; i < BLOCK_I; i++) {
              for (int k = 0; k < d; k++) {
                float val = 0.f;
                for (int j = 0; j < BLOCK_J; j++) {
                  val += Pij[i * BLOCK_J + j] * Vj[j * d + k];
                }
                Oi[i * d + k] = (
                  li[i] * Oi[i * d + k] + val
                ) / lnew[i];
              }
            }

            for (int i = 0; i < BLOCK_I; i++) {
              for (int k = 0; k < d; k++) {
                O[step + (ii + i) * d + k] = Oi[i * d + k];
              }
              l[ii + i] = lnew[i];
            }
          }
        }
      }
    }
    // -------- YOUR CODE HERE  -------- //


    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
