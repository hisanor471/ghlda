#define _USE_MATH_DEFINES
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <omp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/eigen.h>
#include "fastapprox/sse.h"
#include "fastapprox/fastlog.h"
#include "fastapprox/fastgamma.h"

#define MATH_PI           3.141592653589793238462643383279502884197169399375105820974
#define MATH_PI_2         1.570796326794896619231321691639751442098584699687552910487
#define MATH_2_PI         0.636619772367581343075535053490057448137838582961825794990
#define MATH_PI2          9.869604401089358618834490999876151135313699407240790626413
#define MATH_PI2_2        4.934802200544679309417245499938075567656849703620395313206
#define MATH_SQRT1_2      0.707106781186547524400844362104849039284835937688474036588
#define MATH_SQRT_PI_2    1.253314137315500251207882642405522626503493370304969158314
#define MATH_LOG_PI       1.144729885849400174143427351353058711647294812915311571513
#define MATH_LOG_2_PI     -0.45158270528945486472619522989488214357179467855505631739
#define MATH_LOG_PI_2     0.451582705289454864726195229894882143571794678555056317392

// Binder
namespace py = pybind11;// This is to make the code in accord with the pybin11 document
using IntMatrixMap  = std::unordered_map <int, Eigen::MatrixXf>;
using LongMatrixMap = std::unordered_map <long long, Eigen::MatrixXf>;
using IntIntMap     = std::unordered_map <int, std::unordered_map <int, float> >;
PYBIND11_MAKE_OPAQUE(IntMatrixMap);
PYBIND11_MAKE_OPAQUE(LongMatrixMap);
PYBIND11_MAKE_OPAQUE(IntIntMap);
//End Binder//

// Utility Functions
std::vector <std::string>&Split(const std::string&s, char delim, std::vector <std::string>&elems) {
   std::stringstream ss(s);
   std::string       item;
   while (std::getline(ss, item, delim)) {
      elems.push_back(item);
   }
   return(elems);
}

std::vector <std::string> Split(const std::string&s, char delim) {
   std::vector <std::string> elems;
   Split(s, delim, elems);
   return(elems);
}

std::vector <int> Split2IntVector(std::string line) {
   std::vector <std::string> elements0 = Split(line, ',');
   std::vector <int>         elements;
   for (int i = 0; i < int(elements0.size()); ++i) {
      elements.push_back(std::stoi(elements0[i]));
   }
   return(elements);
}

class GHLDA {// Gaussian Hierarchical Latent Dirichlet Allocation
public:
   std::string name;
   GHLDA(const std::string&name0) {
      name = name0;
   }

   // Data and assignments
   Eigen::MatrixXi doc_word_topic, doc_word_topic_test; // Assignment table: doc_id, word_id, path, level
   Eigen::MatrixXf embedding_center, Psi;
   Eigen::MatrixXf embedding_matrix;
   std::unordered_map <int, std::vector <int> > doc2place_vec, doc2place_vec_test;
   std::unordered_map <int, int> doc2path;
   // Variables in LDA, GLDA
   int num_topics = 0;
   int embedding_dimension;
   // Stats
   std::unordered_map <long long, float> word2count;
   std::unordered_map <int, Eigen::MatrixXf> topic2mu, topic2Sigma;
   std::unordered_map <int, float> topic2num, path2num;
   // Variables in GHLDA, HLDA
   std::map <int, std::vector <int> > path_dict;
   std::unordered_map <int, int> topic_dict;
   std::unordered_map <long long, Eigen::MatrixXf> doc2level_vec, word2topic_vec, word2level_vec, word2path_vec;
   int num_path, num_depth, num_docs, num_docs_test;
   // Hyper-parameters
   int level_allocation_type;
   float kappa, nu;
   float hyper_pi, hyper_m;
   Eigen::VectorXf alpha_level_vec;
   float voc, eta_in_hLDA, beta, gamma;
   // debugging
   int stop_hantei = 0;
   float rescale       = 1.0;
   float inner_rescale = 0.0;
   float min_diagonal_val = 0.0;
   std::unordered_map <int, int> topic2level;
   Eigen::MatrixXf zero_mu, zero_Sigma;

   // Probability mass of multivariate t
   float prob_mass_mvt(const Eigen::MatrixXf& x_mvt0,
                       const float&nu_mvt, const Eigen::MatrixXf& mu_mvt0, const Eigen::MatrixXf& Sigma_mvt,
                       const int& log_true, const int& verbose) {
      Eigen::MatrixXf I = Eigen::MatrixXf::Identity(embedding_dimension, embedding_dimension);

      // DEPRECATED: cholesky_stabilizer_identity is basically set to 0.0
      Eigen::MatrixXf Sigma_stable = Sigma_mvt;

      // vectors have to be (size,1)
      Eigen::MatrixXf x_mvt, mu_mvt;
      if (x_mvt0.rows() < x_mvt0.cols()) {
         x_mvt = x_mvt0.transpose();
      }else{
         x_mvt = x_mvt0;
      }
      if (mu_mvt0.rows() < mu_mvt0.cols()) {
         mu_mvt = mu_mvt0.transpose();
      }else{
         mu_mvt = mu_mvt0;
      }

      float log_prob          = 0.0;
      float prob              = 0.0;
      float first_nume        = 0.0;
      float first_denom_one   = 0.0;
      float first_denom_two   = 0.0;
      float first_denom_three = 0.0;
      float first_denom_four  = 0.0;
      float first_denom       = 0.0;
      float first             = 0.0;
      float second_inside     = 0.0;
      float second            = 0.0;
      float log_determinant   = 0.0;

      // Choleskey fast
      std::string method = "llt";

      // first_numerator
      first_nume = std::lgamma((nu_mvt + float(embedding_dimension)) / 2.0);
      if (verbose == 1) {
         std::cout << first_nume << std::endl;
      }
      // first_denominator
      first_denom_one = std::lgamma(nu_mvt / 2.0);//OK
      if (verbose == 1) {
         std::cout << first_denom_one << std::endl;
      }
      first_denom_two = float(embedding_dimension) / 2.0 * std::log(nu_mvt);
      if (verbose == 1) {
         std::cout << first_denom_two << std::endl;
      }
      first_denom_three = float(embedding_dimension) / 2.0 * std::log(M_PI);
      if (verbose == 1) {
         std::cout << first_denom_three << std::endl;
      }

      if (method == "llt") {
         Eigen::LLT <Eigen::MatrixXf> llt;
         llt.compute(Sigma_stable);
         auto& U = llt.matrixL();
         for (unsigned i = 0; i < Sigma_mvt.rows(); ++i) {
            if (verbose == 1) {
               std::cout << " U " << U(i, i) << std::endl;
            }
            if (min_diagonal_val > 0.0) {
               if (U(i, i) <= 0.0) {
                  log_determinant += std::log(min_diagonal_val);
               }else{
                  log_determinant += std::log(U(i, i));
               }
            }
         }
         first_denom_four = log_determinant;
         if (verbose == 1) {
            std::cout << first_denom_four << std::endl;
         }
         // second term
         second_inside = std::log(1.0 + (llt.matrixL().solve(x_mvt - mu_mvt)).squaredNorm() / nu_mvt);
         if (verbose == 1) {
            std::cout << second_inside << std::endl;
         }
         second = -((nu_mvt + float(embedding_dimension)) / 2.0) * second_inside;
         if (verbose == 1) {
            std::cout << second << std::endl;
         }
      }else{
      }
      // Additional Calculation
      first_denom = first_denom_one + first_denom_two + first_denom_three + first_denom_four;
      first       = first_nume - first_denom;
      log_prob    = first + second;
      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      return(prob);
   }

   // LDA START //
   void lda_calc_tables_parameters_from_assign() {
      if (num_topics == 0 || embedding_dimension == 0) {
         std::cout << "Please set num_topics and embedding_dimension" << std::endl;
      }else{
         topic2num.clear();
         word2topic_vec.clear();
         doc2topic_vec.clear();
         for (int i = 0; i < doc_word_topic.rows(); ++i) {
            int doc   = doc_word_topic(i, 0);
            int word  = doc_word_topic(i, 1);
            int topic = doc_word_topic(i, 2);
            //Eigen::MatrixXf x = embedding_matrix.row(word);
            // Create tables
            if (doc2topic_vec.find(doc) == doc2topic_vec.end()) {
               doc2topic_vec[doc] = Eigen::MatrixXf::Zero(num_topics, 1);
            }
            doc2topic_vec[doc](topic, 0) += 1.0;

            // Create Parameters
            if (topic2num.find(topic) == topic2num.end()) {
               topic2num[topic] = 1.0;
            }else{
               topic2num[topic] += 1.0;
            }
            if (word2topic_vec.find(word) == word2topic_vec.end()) {
               word2topic_vec[word] = Eigen::MatrixXf::Zero(num_topics, 1);
            }
            word2topic_vec[word](topic, 0) += 1.0;
         }
      }
   }

   void lda_collapsed_gibbs_sample_parallel(const int num_iteration, const int parallel_loop, int num_threads) {
      stop_hantei = 0;

      if (num_threads < 1) {
         num_threads = omp_get_max_threads();
         std::cout << "Using " << num_threads << " threads" << std::endl;
      }
      omp_set_num_threads(num_threads);

      const gsl_rng_type *T;
      gsl_rng **          r;
      gsl_rng_env_setup();
      T = gsl_rng_mt19937;
      r = (gsl_rng **)malloc(num_threads * sizeof(gsl_rng *));
      std::random_device rand_device;
      std::mt19937       mt(rand_device());
      for (int i = 0; i < num_threads; i++) {
         r[i] = gsl_rng_alloc(T);
         gsl_rng_set(r[i], mt());
      }

      for (int itr = 0; itr < num_iteration; ++itr) {
         if (stop_hantei == 1) {
            std::cout << "ERROR: COULD NOT SAMPLE TOPIC" << std::endl;
            break;
         }

         Eigen::MatrixXi doc_word_topic_update = Eigen::MatrixXi::Zero(parallel_loop, 4);
         Eigen::MatrixXf debug = Eigen::MatrixXf::Zero(parallel_loop, 7);

         std::unordered_map <int, std::vector <float> > debug2;

         #pragma omp parallel for
         for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
            int thread_id = omp_get_thread_num();
            int place     = int(floor(float(doc_word_topic.rows()) * gsl_rng_uniform(r[thread_id])));
            int doc       = doc_word_topic(place, 0);
            int word      = doc_word_topic(place, 1);

            // Create super topic ratio
            std::vector <float> topic_ratio;
            std::vector <float> word_ratio, ratio;

            //create sub topic ratio
            for (int i = 0; i < num_topics; ++i) {
               float temp = alpha_topic_vec(i) + doc2topic_vec[doc](i, 0);
               topic_ratio.push_back(temp);
            }

            // create word ratio
            float agg_topic = 0;
            float max_val   = -9999999999999;
            float min_val   = 9999999999999;
            std::unordered_map <int, float> word2agg;
            float beta_agg = beta * voc;

            for (int i = 0; i < num_topics; ++i) {
               float temp = 0.0;
               temp = word2topic_vec[word](i, 0) + beta;
               word_ratio.push_back(temp);
               word2agg[i] = beta_agg + topic2num[i];
            }

            for (int i = 0; i < num_topics; ++i) {
               // calc prob
               float temp = std::log(topic_ratio[i]) + std::log(word_ratio[i]) - std::log(word2agg[i]);
               ratio.push_back(temp);
               if (max_val < temp) {
                  max_val = temp;
               }
               if (min_val > temp) {
                  min_val = temp;
               }
            }

            for (int i = 0; i < int(ratio.size()); ++i) {
               ratio[i]   = ratio[i] - max_val;
               ratio[i]   = std::exp(ratio[i]);
               agg_topic += ratio[i];
            }

            // sample
            double prob[ratio.size()]; // Probability array
            for (int i = 0; i < int(ratio.size()); ++i) {
               prob[i] = double(ratio[i] / agg_topic);
            }
            unsigned int mult_op[ratio.size()];
            int          num_sample = 1;
            gsl_ran_multinomial(r[thread_id], ratio.size(), num_sample, prob, mult_op);
            int new_topic = -1;
            for (int i = 0; i < int(ratio.size()); ++i) {
               if (mult_op[i] == 1) {
                  new_topic = i;
                  break;
               }
            }

            #pragma omp critical
            {
               doc_word_topic_update(itr_inner, 0) = place;
               doc_word_topic_update(itr_inner, 1) = new_topic;
               debug(itr_inner, 0) = agg_topic;
               debug(itr_inner, 1) = max_val;
               debug(itr_inner, 2) = min_val;
               debug2[itr_inner]   = ratio;
            }
         }

         // Update assignments tables parameters
         for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
            int place     = doc_word_topic_update(itr_inner, 0);
            int new_topic = doc_word_topic_update(itr_inner, 1);
            int doc       = doc_word_topic(place, 0);
            int word      = doc_word_topic(place, 1);
            int topic     = doc_word_topic(place, 2);

            if (itr_inner % 5000 == 0) {
               std::cout << "the topic prop is" << std::endl;
               for (int j = 0; j < int((debug2[itr_inner]).size()); ++j) {
                  std::cout << (debug2[itr_inner])[j] << std::endl;
               }
               std::cout << itr_inner << " replace " << place << " old topic " << topic << " new topic " << new_topic << " agg_topic " << debug(itr_inner, 0) << std::endl;
               std::cout << "=========================================" << std::endl;
            }

            if (new_topic < 0) {
               std::cout << "ERROR" << std::endl;
               stop_hantei = 1;
               break;
            }

            // word embedding : x (1,embedding_dimension)
            Eigen::MatrixXf x = embedding_matrix.row(word);
            // Subtract
            doc2topic_vec[doc](topic, 0) = doc2topic_vec[doc](topic, 0) - 1.0;
            word2topic_vec[word](topic, 0) = word2topic_vec[word](topic, 0) - 1.0;
            topic2num[topic] = topic2num[topic] - 1.0;
            // Add
            doc2topic_vec[doc](new_topic, 0) = doc2topic_vec[doc](new_topic, 0) + 1.0;
            word2topic_vec[word](new_topic, 0) = word2topic_vec[word](new_topic, 0) + 1.0;
            topic2num[new_topic] = topic2num[new_topic] + 1.0;
            // Update assignments
            doc_word_topic(place, 2) = new_topic;
         }
      }
      for (int i = 0; i < num_threads; i++) {
         gsl_rng_free(r[i]);
      }
   }

   void lda_evaluate_held_out_log_likelihood(const std::vector <int> doc_test_ids, const int num_iteration, const int burn_in, const int approx, int num_threads, int verbose) { //only approx == 1 ver

      if (num_threads < 1) {
         num_threads = omp_get_max_threads();
         std::cout << "Using " << num_threads << " threads" << std::endl;
      }

      omp_set_num_threads(num_threads);
      int max_num_threads = omp_get_max_threads();
      const gsl_rng_type *T;
      gsl_rng **          r;
      gsl_rng_env_setup();
      T = gsl_rng_mt19937;
      r = (gsl_rng **)malloc(max_num_threads * sizeof(gsl_rng *));
      std::random_device rand_device;
      std::mt19937       mt(rand_device());
      for (int i = 0; i < max_num_threads; i++) {
         r[i] = gsl_rng_alloc(T);
         gsl_rng_set(r[i], mt());
      }

      std::unordered_map <int, int> update_topic;
      for (auto itr = topic2num.begin(); itr != topic2num.end(); ++itr) {
         update_topic[itr->first] = 1;
      }

      #pragma omp parallel for
      for (int k = 0; k < int(doc_test_ids.size()); ++k) {
         int             doc_test = doc_test_ids[k];
         int             doc_size = int(doc2place_vec_test[doc_test].size());
         Eigen::MatrixXi doc_word_topic_thread = Eigen::MatrixXi::Zero(doc_size, 8);  // take log sum of this
         int             cnt = 0;
         for (int i = 0; i < doc_size; ++i) {
            int place = doc2place_vec_test[doc_test][i];
            doc_word_topic_thread(cnt, 0) = doc_word_topic_test(place, 0);
            doc_word_topic_thread(cnt, 1) = doc_word_topic_test(place, 1);
            doc_word_topic_thread(cnt, 2) = doc_word_topic_test(place, 2);
            cnt += 1;
         }
         Eigen::MatrixXf heldout_prob = Eigen::MatrixXf::Zero(doc_word_topic_thread.rows(), 1);

         for (int up_to = 0; up_to < int(doc_word_topic_thread.rows()); ++up_to) {
            if (verbose == 1) {
               std::cout << "up_to:" << up_to << std::endl;
            }
            int             thread_id     = omp_get_thread_num();
            Eigen::MatrixXf topic_vec_sub = Eigen::MatrixXf::Zero(num_topics, 1);
            Eigen::MatrixXi doc_word_topic_thread_temp = Eigen::MatrixXi::Zero(up_to + 1, 10);

            if (approx == 1) {
               for (int i = 0; i < (up_to + 1); ++i) {
                  doc_word_topic_thread_temp(i, 0) = doc_word_topic_thread(i, 0);
                  doc_word_topic_thread_temp(i, 1) = doc_word_topic_thread(i, 1);
                  doc_word_topic_thread_temp(i, 2) = doc_word_topic_thread(i, 2);
                  int topic = doc_word_topic_thread_temp(i, 2);
                  topic_vec_sub(topic, 0) += 1.0;
               }
               float agg   = 0;
               float count = 0;
               for (int rep = 0; rep < num_iteration; ++rep) {
                  //for (int place = 0; place < int(doc_word_topic_thread_temp.rows()); ++place) {
                  int place = int(doc_word_topic_thread_temp.rows()) - 1;
                  int word      = doc_word_topic_thread_temp(place, 1);
                  int old_topic = doc_word_topic_thread_temp(place, 2);
                  std::vector <float> topic_ratio;
                  std::vector <float> ratio;
                  for (int i = 0; i < num_topics; ++i) {
                      float temp = alpha_topic_vec(i) + topic_vec_sub(i, 0);
                      topic_ratio.push_back(temp);
                  }
                  // create ratio
                  float max_val = -9999999999999;
                  std::unordered_map <int, float> word_prob, word2agg;
                  if (word2topic_vec.find(word) == word2topic_vec.end()) {
                      for (int i = 0; i < num_topics; i++) {
                          word_prob[i] = beta;
                          word2agg[i]  = voc * beta + topic2num[i];
                      }
                  }else{
                      for (int i = 0; i < num_topics; i++) {
                          word_prob[i] = word2topic_vec[word](i, 0) + beta;
                          word2agg[i]  = voc * beta + topic2num[i];
                      }
                  }
                  for (int i = 0; i < num_topics; ++i) {
                      float temp = std::log(topic_ratio[i]) + word_prob[i];
                      ratio.push_back(temp);
                      if (max_val < temp) {
                          max_val = temp;
                      }
                  }
                  float agg_topic = 0;
                  for (int i = 0; i < int(ratio.size()); ++i) {
                      ratio[i]   = ratio[i] - max_val;
                      ratio[i]   = std::exp(ratio[i]);
                      agg_topic += ratio[i];
                  }
                  // sample
                  double prob[ratio.size()];                // Probability array
                  for (int i = 0; i < int(ratio.size()); ++i) {
                      prob[i] = double(ratio[i] / agg_topic);
                  }
                  unsigned int mult_op[ratio.size()];
                  int          num_sample = 1;
                  gsl_ran_multinomial(r[thread_id], ratio.size(), num_sample, prob, mult_op);
                  int new_topic = -1;
                  for (int i = 0; i < int(ratio.size()); ++i) {
                      if (mult_op[i] == 1) {
                          new_topic = i;
                          break;
                      }
                  }
                  topic_vec_sub(old_topic, 0)         -= 1.0;
                  topic_vec_sub(new_topic, 0)         += 1.0;
                  doc_word_topic_thread_temp(place, 2) = new_topic;
                  if (rep > burn_in) {
                      for (int i = 0; i < num_topics; ++i) {
                          if (word2topic_vec.find(word) != word2topic_vec.end()) {
                              agg += prob[i] * (word2topic_vec[word](i, 0) + beta) / (voc * beta + topic2num[i]);
                          }else{
                              agg += prob[i] * beta / (voc * beta + topic2num[i]);
                          }
                          count += 1.0;
                      }
                  }//if (rep > burn_in) {
                  float prob_2 = agg / count;
                  heldout_prob(up_to, 0) = prob_2;
               }             //for(int rep = 0;rep < num_iteration;++rep){
            }  //if(approx == 1)
         }  //for (int up_to = 0; up_to < int(doc_word_topic_sub.rows()); ++up_to) {
         #pragma omp critical
         {
            doc_test2heldout_prob[doc_test] = heldout_prob;
            std::cout << "Finished: " << doc_test << std::endl;
         } //#pragma omp critical
      }    //for(int k = 0;k < int(test_doc_ids.size());
   }

   // LDA: END //

   // HLDA-fixed: START
   std::unordered_map <int, std::unordered_map <int, float> > topic2word_map;
   std::unordered_map <int, float> zero_topic2word_map;
   std::unordered_map <int, float> level2eta_in_HLDA;

   float hlda_posterior_predictive_robust(const int& doc,
                                          const std::vector <int>& topic_vec,
                                          const int& level,
                                          std::unordered_map <int, float>& topic2num_temp,
                                          std::unordered_map <int, std::unordered_map <int, float> >& topic2word_map_temp,
                                          std::unordered_map <long long, float>& words_in_doc_thread,
                                          std::unordered_map <int, float>& topic2third_nume_inside,
                                          const int& log_true, const int& verbose) {
      std::vector <int> place_vec = doc2place_vec[doc];
      std::vector <int> place_level_vec;
      for (int i = 0; i < int(place_vec.size()); ++i) {
         if (doc_word_topic(place_vec[i], 3) == level) {
            place_level_vec.push_back(place_vec[i]);
         }
      }

      float eta_in_hLDA = level2eta_in_HLDA[level];
      const int topic = topic_vec[level];
      float n = 0;
      float m = 0;
      n = float(topic2num_temp[topic]);
      m = float(place_level_vec.size());

      float log_prob     = 0.0;
      float prob         = 0.0;
      float first_nume   = 0.0;
      float first_denom  = 0.0;
      float second_nume  = 0.0;
      float second_denom = 0.0;

      first_nume = std::lgamma(n + voc * eta_in_hLDA);
      std::unordered_map <int, float> word_map_temp = topic2word_map_temp[topic];

      for (auto itr = word_map_temp.begin(); itr != word_map_temp.end(); ++itr) {
         float temp = word_map_temp[itr->first];
         first_denom = first_denom + std::lgamma(temp + eta_in_hLDA);

         float temp2 = word_map_temp[itr->first];
         if (words_in_doc_thread.find(itr->first) != words_in_doc_thread.end()) {
            for (int i = 0; i < int(place_level_vec.size()); ++i) {
               if (doc_word_topic(place_level_vec[i], 1) == itr->first) {
                  temp2 = temp2 + 1.0;
               }
            }
         }
         second_nume = second_nume + std::lgamma(temp2 + eta_in_hLDA);
      }

      second_denom = std::lgamma(n + m + voc * eta_in_hLDA);
      log_prob = first_nume - first_denom + second_nume - second_denom;
      log_prob = log_prob - inner_rescale;

      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      if (verbose == 1) {
         std::cout << " ========================================= " << std::endl;
         std::cout << " topic " << topic << std::endl;
         std::cout << "  n " << n << std::endl;
         std::cout << "  m " << m << std::endl;
         std::cout << n + voc * eta_in_hLDA << "  first_nume " << first_nume << std::endl;
         std::cout << "  first_denom " << first_denom << std::endl;
         std::cout << "  second_nume " << second_nume << std::endl;
         std::cout << n + m + voc * eta_in_hLDA << "  second_denom " << second_denom << std::endl;
         std::cout << "  log_prob " << prob << std::endl;
      }
      return(prob);
   }

   float hlda_posterior_predictive_robust_rev(const int& doc,
                                          const std::vector <int>& topic_vec,
                                          const int& level,
                                          std::unordered_map <int, float>& topic2num_temp,
                                          std::unordered_map <int, std::unordered_map <int, float> >& topic2word_map_temp,
                                          std::unordered_map <long long, float>& words_in_doc_thread,
                                          std::unordered_map <int, float>& topic2third_nume_inside,
                                          const int& log_true, const int& verbose) {
      std::vector <int> place_vec = doc2place_vec[doc];
      std::vector <int> place_level_vec;
      for (int i = 0; i < int(place_vec.size()); ++i) {
         if (doc_word_topic(place_vec[i], 3) == level) {
            place_level_vec.push_back(place_vec[i]);
         }
      }

      float     eta_in_hLDA = level2eta_in_HLDA[level];
      const int topic       = topic_vec[level];
      float     n           = float(topic2num_temp[topic]);
      float     m           = float(place_level_vec.size());
      float     log_prob    = 0.0;
      float     prob        = 0.0;

      float second_nume_first_denom = 0.0;
      std::unordered_map <int, float> word_map_temp = topic2word_map_temp[topic];
      for (auto itr = word_map_temp.begin(); itr != word_map_temp.end(); ++itr) {
         int   word_0 = itr->first;
         float temp   = itr->second;
         //first_denom = first_denom + std::lgamma(temp + eta_in_hLDA);
         float temp2 = 0;
         for (int i = 0; i < int(place_level_vec.size()); ++i) {
            if (doc_word_topic(place_level_vec[i], 1) == word_0) {
               temp2 = temp2 + 1.0;
            }
         }
         //second_nume = second_nume + std::lgamma(temp2 + temp + eta_in_hLDA);
         for (int jj = 0; jj < int(temp2); ++jj) {
            second_nume_first_denom += std::log(temp + eta_in_hLDA + float(jj));
         }
      }

      //first_nume = std::lgamma(n + voc * eta_in_hLDA);
      //second_denom = std::lgamma(n + m + voc * eta_in_hLDA);
      float first_nume_second_denom = 0.0;
      float second_denom_first_nume = 0.0;
      for (int jj = 0; jj < int(m); ++jj) {
         second_denom_first_nume += std::log(n + voc * eta_in_hLDA + float(jj));
      }
      first_nume_second_denom = -second_denom_first_nume;

      //log_prob = first_nume - second_denom - first_denom + second_nume ;
      log_prob = first_nume_second_denom + second_nume_first_denom;

      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      return(prob);
   }

   float hlda_posterior_predictive_single_robust(const int& place,
                                                 const int& level,
                                                 const int& topic,
                                                 std::unordered_map <int, std::unordered_map <int, float> >& topic2word_map_temp,
                                                 const int& log_true, const int& verbose) {
      float eta_in_HLDA = level2eta_in_HLDA[level];
      int   word        = doc_word_topic(place, 1);
      float n           = topic2word_map_temp[topic][word];
      float log_prob    = 0.0;
      float prob        = 0.0;
      log_prob = std::log(n + eta_in_HLDA);

      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      return(prob);
   }

   float hlda_posterior_predictive_single_robust_rev(const int& place,
                                                     const int& level,
                                                     const int& topic,
                                                     std::unordered_map <int, float>& topic2num_temp,
                                                     std::unordered_map <int, std::unordered_map <int, float> >& topic2word_map_temp,
                                                     const int& log_true, const int& verbose) {
      float eta_in_hLDA = level2eta_in_HLDA[level];

      //std::vector<int> place_vec = doc2place_vec[doc];
      std::vector <int> place_level_vec;
      place_level_vec.push_back(place);

      // KOKOGA MONDAI
      float n        = float(topic2num_temp[topic]);
      float m        = 1;
      float log_prob = 0.0;
      float prob     = 0.0;

      //float first_denom_second_nume = 0.0;
      float second_nume_first_denom = 0.0;
      //if (topic2word_map_temp.find(topic) != topic2word_map_temp.end()) {
      std::unordered_map <int, float> word_map_temp = topic2word_map_temp[topic];
      for (auto itr = word_map_temp.begin(); itr != word_map_temp.end(); ++itr) {
         int   word_0 = itr->first;
         float temp   = itr->second;
         //first_denom = first_denom + std::lgamma(temp + eta_in_hLDA);
         float temp2 = 0;
         for (int i = 0; i < int(place_level_vec.size()); ++i) {
            if (doc_word_topic(place_level_vec[i], 1) == word_0) {
               temp2 = temp2 + 1.0;
            }
         }
         //second_nume = second_nume + std::lgamma(temp2 + temp + eta_in_hLDA);
         for (int jj = 0; jj < int(temp2); ++jj) {
            second_nume_first_denom += std::log(temp + eta_in_hLDA + float(jj));
         }
      }

      //first_nume = std::lgamma(n + voc * eta_in_hLDA);
      //second_denom = std::lgamma(n + m + voc * eta_in_hLDA);
      float first_nume_second_denom = 0.0;
      float second_denom_first_nume = 0.0;
      for (int jj = 0; jj < m; ++jj) {
         second_denom_first_nume += std::log(n + voc * eta_in_hLDA + float(jj));
      }
      first_nume_second_denom = -second_denom_first_nume;

      //log_prob = first_nume - second_denom - first_denom + second_nume ;
      log_prob = first_nume_second_denom + second_nume_first_denom;

      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      return(prob);
   }

   void hlda_calc_tables_parameters_from_assign() {
      if (1 == 1) {
         doc2level_vec.clear();
         word2count.clear();
         topic2word_map.clear();
         for (auto itr = topic2num.begin(); itr != topic2num.end(); ++itr) {
            topic2num[itr->first]      = 0.0;
            topic2word_map[itr->first] = zero_topic2word_map;
         }
         topic2num[-1]      = 0.0;
         topic2word_map[-1] = zero_topic2word_map;

         for (int i = 0; i < doc_word_topic.rows(); ++i) {
            int doc   = doc_word_topic(i, 0);
            int word  = doc_word_topic(i, 1);
            int level = doc_word_topic(i, 3);
            int topic = doc_word_topic(i, 4);

            // Create tables
            if (doc2level_vec.find(doc) == doc2level_vec.end()) {
               doc2level_vec[doc] = level_gamma * Eigen::MatrixXf::Ones(num_depth + 1000, 1);
            }
            doc2level_vec[doc](level, 0) += 1.0;
            if (word2count.find(word) == word2count.end()) {
               word2count[word] = 1.0;
            }else{
               word2count[word] += 1.0;
            }
            if (topic2word_map.find(topic) == topic2word_map.end()) {
               topic2word_map[topic] = zero_topic2word_map;
            }
            if (topic2word_map[topic].find(word) == topic2word_map[topic].end()) {
               topic2word_map[topic][word] = 1.0;
            }else{
               topic2word_map[topic][word] += 1.0;
            }
            num_not_assigned = 0.0;
            if (topic == -1) {
               num_not_assigned += 1.0;
            }
            if (topic != -1) {
               // Create Parameters
               if (topic2num.find(topic) == topic2num.end()) {
                  topic2num[topic] = 1.0;
               }else{
                  topic2num[topic] += 1.0;
               }
            }
         }
         std::cout << "hlda_calc_tables finished" << std::endl;
      }
   }

   void hlda_update_parameters_subtract(const int topic, const int word) { // Is it ok?
      float N0 = 0.0;

      if (topic2num.find(topic) != topic2num.end()) {
         N0 = topic2num[topic];
      }
      if (N0 < 1.1) {
         topic2num[topic]      = 0.0;
         topic2word_map[topic] = zero_topic2word_map;
      }else{
         topic2num[topic] = N0 - 1.0;
      }
      float N1 = 0.0;
      if (topic2word_map.find(topic) != topic2word_map.end()) {
         N1 = topic2word_map[topic][word];
      }else{
         topic2word_map[topic] = zero_topic2word_map;
      }
      if (N1 < 1.1) {
         topic2word_map[topic][word] = 0.0;
      }else{
         topic2word_map[topic][word] = N1 - 1.0;
      }
   }

   void hlda_update_parameters_subtract_thread(const int topic, const int word,
                                               std::unordered_map <int, float>& topic2num_temp,
                                               std::unordered_map <int, std::unordered_map <int, float> >& topic2word_map_temp,
                                               int& hantei) {
      float N0 = 0.0;

      if (topic2num_temp.find(topic) != topic2num_temp.end()) {
         N0 = topic2num_temp[topic];
      }
      if (N0 < 1.1) {
         topic2num_temp[topic]      = 0.0;
         topic2word_map_temp[topic] = zero_topic2word_map;
         hantei = 0;
      }else{
         topic2num_temp[topic] = N0 - 1.0;
      }
      float N1 = 0.0;
      if (topic2word_map_temp.find(topic) != topic2word_map_temp.end()) {
         N1 = topic2word_map_temp[topic][word];
      }else{
         topic2word_map_temp[topic] = zero_topic2word_map;
         hantei = 0;
      }
      if (N1 < 1.1) {
         topic2word_map_temp[topic][word] = 0.0;
         hantei = 0;
      }else{
         topic2word_map_temp[topic][word] = N1 - 1.0;
      }
   }

   void hlda_update_parameters_add(const int& new_topic, const int word) {
      if (topic2num.find(new_topic) != topic2num.end()) {
         float N3 = topic2num[new_topic];
         topic2num[new_topic] = N3 + 1.0;
      }else{
         topic2num[new_topic] = 1.0;
      }
      if (topic2word_map.find(new_topic) != topic2word_map.end()) {
         float N4 = topic2word_map[new_topic][word];
         topic2word_map[new_topic][word] = N4 + 1.0;
      }else{
         topic2word_map[new_topic]        = zero_topic2word_map;
         topic2word_map[new_topic][word] += 1.0;
      }
   }

   void hlda_update_parameters_add_thread(const int new_topic, const int word,
                                          std::unordered_map <int, float>& topic2num_temp,
                                          std::unordered_map <int, std::unordered_map <int, float> >& topic2word_map_temp) {
      if (topic2num_temp.find(new_topic) != topic2num_temp.end()) {
         float N0 = topic2num_temp[new_topic];
         topic2num_temp[new_topic] = N0 + 1.0;
      }else{
         topic2num_temp[new_topic] = 1.0;
      }
      if (topic2word_map_temp[new_topic].find(word) != topic2word_map_temp[new_topic].end()) {
         float N1 = topic2word_map_temp[new_topic][word];
         topic2word_map_temp[new_topic][word] = N1 + 1.0;
      }else{
         topic2word_map_temp[new_topic]       = zero_topic2word_map;
         topic2word_map_temp[new_topic][word] = 1.0;
      }
   }

   void hlda_fixed_collapsed_gibbs_sample_parallel(const int num_iteration, const int parallel_loop, int num_threads, const int verbose) {
      if (verbose == 1) {
         std::cout << "Enter hlda_collapsed_gibbs_sample_parallel" << std::endl;
      }
      stop_hantei = 0;
      if (num_threads < 1) {
         num_threads = omp_get_max_threads();
         std::cout << "Using " << num_threads << " threads" << std::endl;
      }
      omp_set_num_threads(num_threads);

      const gsl_rng_type *T;
      gsl_rng **          r;
      gsl_rng_env_setup();
      T = gsl_rng_mt19937;
      r = (gsl_rng **)malloc(num_threads * sizeof(gsl_rng *));
      std::random_device rand_device;
      std::mt19937       mt(rand_device());
      for (int i = 0; i < num_threads; i++) {
         r[i] = gsl_rng_alloc(T);
         gsl_rng_set(r[i], mt());
      }

      for (int itr_outer = 0; itr_outer < num_iteration; ++itr_outer) {
         if (stop_hantei == 1) {
            std::cout << "ERROR: COULD NOT SAMPLE TOPIC" << std::endl;
            break;
         }

         #pragma omp parallel for
         for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
            if (stop_hantei != 1) {
               int thread_id   = omp_get_thread_num();
               int doc_thread  = int(floor(float(num_docs) * gsl_rng_uniform(r[thread_id])));
               int path_thread = doc2path[doc_thread];

               //Used in path sampling
               std::unordered_map <int, float> path2num_thread  = path2num;
               std::unordered_map <int, float> topic2num_thread = topic2num;
               std::unordered_map <int, std::unordered_map <int, float> > topic2word_map_thread = topic2word_map;
               //Used in level sampling
               std::unordered_map <int, float> topic2num_thread_2 = topic2num;
               std::unordered_map <int, std::unordered_map <int, float> > topic2word_map_thread_2 = topic2word_map;
               std::unordered_map <long long, float> words_in_doc_thread;

               if (1 == 1) {
                  //if(path2num_thread[path_thread]>0){
                  // Subtract Path
                  path2num_thread[path_thread] = path2num_thread[path_thread] - 1.0;

                  // Minus document assignment
                  std::vector <int> place_vec_thread = doc2place_vec[doc_thread];
                  for (int i = 0; i < int(place_vec_thread.size()); ++i) {
                     int alter_word  = doc_word_topic(place_vec_thread[i], 1);
                     int alter_topic = doc_word_topic(place_vec_thread[i], 4);
                     int hantei_temp = 1;
                     hlda_update_parameters_subtract_thread(alter_topic, alter_word, topic2num_thread, topic2word_map_thread, hantei_temp);
                     if (words_in_doc_thread.find(alter_word) == words_in_doc_thread.end()) {
                        words_in_doc_thread[alter_word] = 1.0;
                     }
                  }// for(int i=0;i<int(place_vec_thread.size());++i){

                  if (verbose == 1) {
                     std::cout << "After hlda_update_parameters_subtract_thread" << std::endl;
                  }

                  std::map <int, std::vector <int> > new_path_dict = path_dict;

                  if (verbose == 1) {
                     for (auto itr = new_path_dict.begin(); itr != new_path_dict.end(); ++itr) {
                        std::vector <int> v1          = itr->second;
                        std::string       temp_string = std::to_string(itr->first) + ":";
                        for (int i = 0; i < int(v1.size()); ++i) {
                           temp_string += std::to_string(v1[i]) + ",";
                        }
                        std::cout << temp_string << std::endl;
                     }
                  }//if(verbose == 1){

                  // Step A: sample path for a document
                  std::string                     prob_string = "";
                  float                           agg_topic   = 0;
                  float                           max_val     = -9999999999999;
                  std::vector <int>               path_id_vec;
                  std::vector <float>             ratio;
                  std::vector <std::string>       topic_string_vec;
                  std::unordered_map <int, float> topic2third_nume_inside;

                  for (auto itr = new_path_dict.begin(); itr != new_path_dict.end(); ++itr) {
                     int     path_id         = itr->first;
                     path_id_vec.push_back(path_id);
                     std::vector <int> topic_vec = itr->second;
                     float             temp      = 0.0;
                     float             temp_in   = 0.0;
                     for (int j = 0; j < int(topic_vec.size()); ++j) {
                        temp_in = hlda_posterior_predictive_robust_rev(doc_thread,
                                                                   topic_vec, j, topic2num_thread, topic2word_map_thread, words_in_doc_thread, topic2third_nume_inside, 1, verbose);
                        temp += temp_in;
                     }
                     if (path2num_thread.find(path_id) != path2num_thread.end()) {
                        temp += std::log(path2num_thread[path_id] + path_gamma);
                     }else{
                        temp += std::log(path_gamma);
                     }
                     ratio.push_back(temp);
                     if (max_val < temp) {
                        max_val = temp;
                     }
                  }

                  if (verbose == 1) {
                     std::cout << "After Step A before sample" << std::endl;
                  }

                  // sample
                  prob_string = "";
                  for (int i = 0; i < int(ratio.size()); ++i) {
                     ratio[i]    = ratio[i] - max_val;
                     ratio[i]    = std::exp(ratio[i]);
                     agg_topic  += ratio[i];
                  }
                  double prob[int(ratio.size())]; // Probability array
                  for (int i = 0; i < int(ratio.size()); ++i) {
                     prob[i] = double(ratio[i] / agg_topic);
                  }
                  unsigned int mult_op[ratio.size()];
                  int          num_sample = 1;
                  gsl_ran_multinomial(r[thread_id], int(ratio.size()), num_sample, prob, mult_op);
                  int new_path = -1;
                  for (int i = 0; i < int(ratio.size()); ++i) {
                     if (mult_op[i] == 1) {
                        new_path = path_id_vec[i];
                        break;
                     }
                  }
                  if (new_path < 0) {
                     #pragma omp critical
                     {
                        stop_hantei = 1;
                        std::cout << "ERROR STOP PATH" << std::endl;
                     }
                  }// if(new_path < 0){

                  // new_path
                  std::vector <int> new_path_vec = new_path_dict[new_path];

                  // Step B: sample level
                  if (stop_hantei != 1) {
                     // COPY
                     std::unordered_map <long long, Eigen::MatrixXf> doc2level_vec_thread = doc2level_vec;
                     std::map <int, std::vector <int> > new_place_path_level;
                     std::random_device seed_gen;
                     std::mt19937       engine(seed_gen());
                     std::shuffle(place_vec_thread.begin(), place_vec_thread.end(), engine);
                     int inner_count = 0;

                     // for each word position
                     for (int itr_place = 0; itr_place < int(place_vec_thread.size()); ++itr_place) {
                        int place     = place_vec_thread[itr_place];
                        int word      = doc_word_topic(place, 1);
                        int old_level = doc_word_topic(place, 3);
                        int old_topic = doc_word_topic(place, 4);

                        int hantei_update_topic2word_map = 1;
                        hlda_update_parameters_subtract_thread(old_topic, word, topic2num_thread_2, topic2word_map_thread_2, hantei_update_topic2word_map);
                        int hantei_update_doc2level = 0;
                        if (doc2level_vec_thread[doc_thread](old_level, 0) > 0) {
                           doc2level_vec_thread[doc_thread](old_level, 0) = doc2level_vec_thread[doc_thread](old_level, 0) - 1.0;
                           hantei_update_doc2level = 1;
                        }

                        // Create level topic ratio
                        std::vector <float> level_ratio;
                        float agg_level = 0.0;
                        if (level_allocation_type == 0) {// Pure LDA
                           for (int i = 0; i < int(new_path_vec.size()); ++i) {
                              float temp = alpha_level_vec(i) + doc2level_vec_thread[doc_thread](i, 0);
                              level_ratio.push_back(temp);
                              agg_level += temp;
                           }
                        }else{// HLDA
                           for (int i = 0; i < int(new_path_vec.size()); ++i) {
                              float first       = 1.0;
                              float first_nume  = hyper_m * hyper_pi + doc2level_vec_thread[doc_thread](i, 0);
                              float first_denom = hyper_pi;
                              for (int j = i; j < int(new_path_vec.size()); ++j) {
                                 first_denom += doc2level_vec_thread[doc_thread](j, 0);
                              }
                              first = first_nume / first_denom;
                              float second = 1.0;
                              for (int j = 0; j < i; ++j) {
                                 float second_nume_inside = 0.0;
                                 for (int k = j + 1; k < int(new_path_vec.size()); ++k) {
                                    second_nume_inside += doc2level_vec_thread[doc_thread](k, 0);
                                 }
                                 float second_nume         = (1.0 - hyper_m) * hyper_pi + second_nume_inside;
                                 float second_denom_inside = 0.0;
                                 for (int k = j; k < int(new_path_vec.size()); ++k) {
                                    second_denom_inside += doc2level_vec_thread[doc_thread](k, 0);
                                 }
                                 float second_denom = hyper_pi + second_denom_inside;
                                 second = second * (second_nume / second_denom);
                              }
                              float temp = first * second;
                              level_ratio.push_back(temp);
                              agg_level += temp;
                           }
                        }//if(level_allocation_type == 0){}else{}

                        float agg_topic = 0;
                        max_val = -9999999999999;
                        std::vector <float> ratio;
                        for (int j = 0; j < int(new_path_vec.size()); ++j) {
                           int     topic            = new_path_vec[j];
                           float   log_prob_a       = std::log(level_ratio[j]) - std::log(agg_level);
                           float log_prob_b = hlda_posterior_predictive_single_robust_rev(place, j, topic,
                                                               topic2num_thread_2, topic2word_map_thread_2, 1, 0);
                           float log_prob = log_prob_a + log_prob_b;
                           ratio.push_back(log_prob);
                           if (max_val < ratio[j]) {
                              max_val = ratio[j];
                           }
                        }
                        agg_topic = 0;
                        for (int j = 0; j < int(ratio.size()); ++j) {
                           ratio[j]   = ratio[j] - max_val;
                           ratio[j]   = std::exp(ratio[j]);
                           agg_topic += ratio[j];
                        }
                        // sample
                        double prob[ratio.size()]; // Probability array
                        for (int j = 0; j < int(ratio.size()); ++j) {
                           prob[j] = double(ratio[j] / agg_topic);
                        }
                        unsigned int mult_op[int(ratio.size())];
                        int          num_sample = 1;
                        gsl_ran_multinomial(r[thread_id], int(ratio.size()), num_sample, prob, mult_op);
                        int new_level = -1;
                        int new_topic = -1;
                        for (int j = 0; j < int(ratio.size()); ++j) {
                           if (mult_op[j] == 1) {
                              new_level = j;
                              new_topic = new_path_vec[new_level];
                              break;
                           }
                        }
                        hlda_update_parameters_add_thread(old_topic, word, topic2num_thread_2, topic2word_map_thread_2);
                        if (hantei_update_doc2level == 1) {
                           doc2level_vec_thread[doc_thread](new_level, 0) = doc2level_vec_thread[doc_thread](new_level, 0) + 1.0;
                        }
                        std::vector <int> result;
                        result.push_back(place);
                        result.push_back(new_path);
                        result.push_back(new_level);
                        result.push_back(new_topic);
                        new_place_path_level[inner_count] = result;
                        inner_count += 1;
                     }//end sample level

                     if (verbose == 1) {
                        std::cout << "End Step B" << std::endl;
                     }

                     //int discard_flag = 0;
                     #pragma omp critical
                     {
                        if (1 == 1) {
                           sample_counter += 1.0;
                           int hantei     = 1;
                           int change_doc = -1;

                           for (auto itr2 = new_place_path_level.begin(); itr2 != new_place_path_level.end(); ++itr2) {
                              std::vector <int> result = itr2->second;
                              int change_place         = result[0];
                              change_doc = doc_word_topic(change_place, 0);
                              int new_path  = result[1];
                              int new_level = result[2];
                              int new_topic = result[3];
                              int old_path  = doc_word_topic(change_place, 2);
                              int old_level = doc_word_topic(change_place, 3);
                              int old_topic = path_level2topic(old_path, old_level);

                              doc_word_topic(change_place, 2) = new_path;
                              doc_word_topic(change_place, 3) = new_level;
                              doc_word_topic(change_place, 4) = new_topic;

                              doc2level_vec[change_doc](old_level, 0) = doc2level_vec[change_doc](old_level, 0) - 1.0;
                              doc2level_vec[change_doc](new_level, 0) = doc2level_vec[change_doc](new_level, 0) + 1.0;

                              if (1 == 1) {
                                 if (hantei == 1) {    // we only need to do this once
                                    hantei = 0;
                                    path2num[old_path] = path2num[old_path] - 1.0;
                                    path2num[new_path]   = path2num[new_path] + 1.0;
                                    doc2path[change_doc] = new_path;
                                 }//if(hantei==1){

                                 int change_word = doc_word_topic(change_place, 1);
                                 hlda_update_parameters_subtract(old_topic, change_word);
                                 hlda_update_parameters_add(new_topic, change_word);
                              }//if(path2num.find(old_path)!=path2num.end(){}else{}
                           }    //for(auto itr2=new_place_path_level.begin()
                        }       //if(discard_flag == 0){
                     }          //#pragma omp critical
                  }             //if(stop_hantei == 1)
               }                //if(path2num_thread[path_thread]>0)
            }                   //if(stop_hantei == 1)
         }                      //for(int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner)
      }                         //for(int itr = 0; itr < num_iteration; ++itr){
      // free memory
      for (int i = 0; i < num_threads; i++) {
         gsl_rng_free(r[i]);
      }
   }//hlda_collapsed_gibbs_sample_parallel

   // HLDA-Fixed END //

   // HLDA-ncrp START //
   void hlda_ncrp_calc_tables_parameters_from_assign() {
      int verbose = 0;

      if (verbose == 1) {
         std::cout << "Enter hlda_ncrp_calc_tables_parameters_from_assign" << std::endl;
      }

      if (1 == 1) {
         doc2level_vec.clear();
         word2count.clear();
         for (auto itr = topic2num.begin(); itr != topic2num.end(); ++itr) {
            topic2num[itr->first]      = 0.0;
            topic2word_map[itr->first] = zero_topic2word_map;
         }

         topic2num[-1]      = 0.0;
         topic2word_map[-1] = zero_topic2word_map;

         for (int i = 0; i < doc_word_topic.rows(); ++i) {
            int doc   = doc_word_topic(i, 0);
            int word  = doc_word_topic(i, 1);
            int level = doc_word_topic(i, 3);
            int topic = doc_word_topic(i, 4);

            // Create tables
            if (doc2level_vec.find(doc) == doc2level_vec.end()) {
               doc2level_vec[doc] = level_gamma * Eigen::MatrixXf::Ones(num_depth + 1000, 1);
            }
            doc2level_vec[doc](level, 0) += 1.0;

            if (word2count.find(word) == word2count.end()) {
               word2count[word] = 1.0;
            }else{
               word2count[word] += 1.0;
            }

            if (topic2word_map.find(topic) == topic2word_map.end()) {
               topic2word_map[topic] = zero_topic2word_map;
            }
            if (topic2word_map[topic].find(word) == topic2word_map[topic].end()) {
               topic2word_map[topic][word] = 1.0;
            }else{
               topic2word_map[topic][word] += 1.0;
            }

            num_not_assigned = 0.0;
            if (topic == -1) {
               num_not_assigned += 1.0;
            }
            if (topic != -1) {
               // Create Parameters
               if (topic2num.find(topic) == topic2num.end()) {
                  topic2num[topic] = 1.0;
               }else{
                  topic2num[topic] += 1.0;
               }
            }else{
            }
         }// for (int i = 0; i < doc_word_topic.rows(); ++i) {

         // Erase unnecessary THIS IS THE ONLY DIFFERENCE
         std::map <int, int> used_topic_id_t;
         for (auto itr = path_dict.begin(); itr != path_dict.end(); ++itr) {
            std::vector <int> path_vec_t  = itr->second;
            std::string       temp_string = "";
            for (int j = 0; j < int(path_vec_t.size()); ++j) {
               if (used_topic_id_t.find(path_vec_t[j]) == used_topic_id_t.end()) {
                  used_topic_id_t[path_vec_t[j]] = 1;
               }
            }
         }

         if (verbose == 1) {
            std::cout << "End Loop" << std::endl;
         }

         // SAFER VERSION
         std::unordered_map <int, float> topic2num_t;
         std::unordered_map <int, std::unordered_map <int, float> > topic2word_map_t;
         topic2num_t[-1]      = 0.0;
         topic2word_map_t[-1] = zero_topic2word_map;
         for (auto itr = topic2num.begin(); itr != topic2num.end(); ++itr) {
            int topic = itr->first;
            if (itr->second > 0) {
               topic2num_t[topic]      = itr->second;
               topic2word_map_t[topic] = topic2word_map[topic];
            }
         }
         topic2num.clear();
         topic2num = topic2num_t;
         topic2word_map.clear();
         topic2word_map = topic2word_map_t;
         // OLD VERSION
         //for (auto itr = topic2num.begin(); itr != topic2num.end(); ++itr) {
         //   int topic = itr->first;
         //   if (topic != -1 && used_topic_id_t.find(topic) == used_topic_id_t.end()) {
         //     if(topic2num.find(topic)!=topic2num.end()){
         //           topic2num.erase(topic);
         //       }
         //       if(topic2word_map.find(topic)!=topic2word_map.end()){
         //          topic2word_map.erase(topic);
         //       }
         //    }
         //}
      }
      if (verbose == 1) {
         std::cout << "Exit hlda_ncrp_calc_tables_parameters_from_assign" << std::endl;
      }
   }

   float hlda_ncrp_posterior_predictive_robust(const int& doc,
                                               const std::vector <int>& topic_vec,
                                               const int& level,
                                               const int& last_topic,
                                               std::unordered_map <int, float>& topic2num_temp,
                                               std::unordered_map <int, std::unordered_map <int, float> >& topic2word_map_temp,
                                               std::unordered_map <long long, float>& words_in_doc_thread,
                                               const int& log_true, const int& verbose) {
      std::vector <int> place_vec = doc2place_vec[doc];
      std::vector <int> place_level_vec;
      for (int i = 0; i < int(place_vec.size()); ++i) {
         if (last_topic != 1) {
            if (doc_word_topic(place_vec[i], 3) == level) {
               place_level_vec.push_back(place_vec[i]);
            }
         }else{
            if (doc_word_topic(place_vec[i], 3) >= level) {
               place_level_vec.push_back(place_vec[i]);
            }
         }
      }

      float eta_in_hLDA = level2eta_in_HLDA[level];

      const int topic        = topic_vec[level];
      float     n            = float(topic2num_temp[topic]);
      float     m            = float(place_level_vec.size());
      float     log_prob     = 0.0;
      float     prob         = 0.0;
      float     first_nume   = 0.0;
      float     first_denom  = 0.0;
      float     second_nume  = 0.0;
      float     second_denom = 0.0;

      first_nume = std::lgamma(n + voc * eta_in_hLDA);
      //if (topic2word_map_temp.find(topic) != topic2word_map_temp.end()) {
      if (1 == 1) {
         std::unordered_map <int, float> word_map_temp = topic2word_map_temp[topic];
         for (auto itr = word_map_temp.begin(); itr != word_map_temp.end(); ++itr) {
            float temp = word_map_temp[itr->first];
            first_denom = first_denom + std::lgamma(temp + eta_in_hLDA);

            float temp2 = word_map_temp[itr->first];
            if (words_in_doc_thread.find(itr->first) != words_in_doc_thread.end()) {              //the word is surely in the doc.
               for (int i = 0; i < int(place_level_vec.size()); ++i) {
                  if (doc_word_topic(place_level_vec[i], 1) == itr->first) {
                     temp2 = temp2 + 1.0;
                  }
               }
            }
            second_nume = second_nume + std::lgamma(temp2 + eta_in_hLDA);
         }
      }else{
         first_denom = voc * std::lgamma(eta_in_hLDA);
         float temp3 = 0.0;
         for (auto itr = words_in_doc_thread.begin(); itr != words_in_doc_thread.end(); ++itr) {
            for (int i = 0; i < int(place_level_vec.size()); ++i) {
               if (doc_word_topic(place_level_vec[i], 1) == itr->first) {
                  temp3 = temp3 + 1.0;
               }
            }
         }
         second_nume = second_nume + std::lgamma(temp3 + eta_in_hLDA);
      }
      second_denom = std::lgamma(n + m + voc * eta_in_hLDA);

      if (verbose == 1) {
         std::cout << " ========================================= " << std::endl;
         std::cout << "  n " << n << std::endl;
         std::cout << "  m " << m << std::endl;
         std::cout << "  first_nume " << first_nume << std::endl;
         std::cout << "  first_denom " << first_denom << std::endl;
         std::cout << "  second_nume " << second_nume << std::endl;
         std::cout << "  second_denom " << second_denom << std::endl;
      }

      log_prob = first_nume - first_denom + second_nume - second_denom;

      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      return(prob);
   }

   float hlda_ncrp_posterior_predictive_robust_rev(const int& doc,
                                                   const std::vector <int>& topic_vec,
                                                   const int& level,
                                                   const int& last_topic,
                                                   std::unordered_map <int, float>& topic2num_temp,
                                                   std::unordered_map <int, std::unordered_map <int, float> >& topic2word_map_temp,
                                                   const int& log_true, const int& verbose) {
      std::vector <int> place_vec = doc2place_vec[doc];
      std::vector <int> place_level_vec;
      for (int i = 0; i < int(place_vec.size()); ++i) {
         if (last_topic != 1) {
            if (doc_word_topic(place_vec[i], 3) == level) {
               place_level_vec.push_back(place_vec[i]);
            }
         }else{
            if (doc_word_topic(place_vec[i], 3) >= level) {
               place_level_vec.push_back(place_vec[i]);
            }
         }
      }

      float     eta_in_hLDA = level2eta_in_HLDA[level];
      const int topic       = topic_vec[level];
      float     n           = float(topic2num_temp[topic]);
      float     m           = float(place_level_vec.size());
      float     log_prob    = 0.0;
      float     prob        = 0.0;

      float second_nume_first_denom = 0.0;
      std::unordered_map <int, float> word_map_temp = topic2word_map_temp[topic];
      for (auto itr = word_map_temp.begin(); itr != word_map_temp.end(); ++itr) {
         int   word_0 = itr->first;
         float temp   = itr->second;
         //first_denom = first_denom + std::lgamma(temp + eta_in_hLDA);
         float temp2 = 0;
         for (int i = 0; i < int(place_level_vec.size()); ++i) {
            if (doc_word_topic(place_level_vec[i], 1) == word_0) {
               temp2 = temp2 + 1.0;
            }
         }
         //second_nume = second_nume + std::lgamma(temp2 + temp + eta_in_hLDA);
         for (int jj = 0; jj < int(temp2); ++jj) {
            second_nume_first_denom += std::log(temp + eta_in_hLDA + float(jj));
         }
      }

      //first_nume = std::lgamma(n + voc * eta_in_hLDA);
      //second_denom = std::lgamma(n + m + voc * eta_in_hLDA);
      float first_nume_second_denom = 0.0;
      float second_denom_first_nume = 0.0;
      for (int jj = 0; jj < int(m); ++jj) {
         second_denom_first_nume += std::log(n + voc * eta_in_hLDA + float(jj));
      }
      first_nume_second_denom = -second_denom_first_nume;

      //log_prob = first_nume - second_denom - first_denom + second_nume ;
      log_prob = first_nume_second_denom + second_nume_first_denom;

      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      return(prob);
   }

   float check_topic2word_map() {
      float count_all = 0;
      float max_val   = -1;

      for (auto itr = topic2word_map.begin(); itr != topic2word_map.end(); ++itr) {
         int topic = itr->first;
         for (auto itr2 = topic2word_map[topic].begin(); itr2 != topic2word_map[topic].end(); ++itr2) {
            count_all += itr2->second;
            if (max_val < itr2->second) {
               max_val = itr2->second;
            }
         }
      }
      std::cout << "num assigned: " << count_all << std::endl;
      std::cout << "max val: " << max_val << std::endl;
      return(count_all);
   }

   void hlda_ncrp_collapsed_gibbs_sample_parallel(const int num_iteration, const int parallel_loop, int num_threads, const int verbose) {
      if (num_threads < 1) {
         num_threads = omp_get_max_threads();
         std::cout << "Using " << num_threads << " threads" << std::endl;
      }
      omp_set_num_threads(num_threads);

      const gsl_rng_type *T;
      gsl_rng **          r;
      gsl_rng_env_setup();
      T = gsl_rng_mt19937;
      r = (gsl_rng **)malloc(num_threads * sizeof(gsl_rng *));
      std::random_device rand_device;
      std::mt19937       mt(rand_device());
      for (int i = 0; i < num_threads; i++) {
         r[i] = gsl_rng_alloc(T);
         gsl_rng_set(r[i], mt());
      }

      for (int itr_outer = 0; itr_outer < num_iteration; ++itr_outer) {
         if (stop_hantei == 1) {
            std::cout << "ERROR: COULD NOT SAMPLE TOPIC" << std::endl;
            break;
         }

         #pragma omp parallel for
         for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
            if (stop_hantei != 1) {
               int thread_id   = omp_get_thread_num();
               int doc_thread  = int(floor(float(num_docs) * gsl_rng_uniform(r[thread_id])));
               int path_thread = doc2path[doc_thread];

               //Used in path sampling
               std::unordered_map <int, float> path2num_thread  = path2num;
               std::unordered_map <int, float> topic2num_thread = topic2num;
               std::unordered_map <int, std::unordered_map <int, float> > topic2word_map_thread = topic2word_map;
               //Used in level sampling
               std::unordered_map <int, float> topic2num_thread_2 = topic2num;
               std::unordered_map <int, std::unordered_map <int, float> > topic2word_map_thread_2 = topic2word_map;
               std::unordered_map <long long, Eigen::MatrixXf>            doc2level_vec_thread    = doc2level_vec;

               if (path2num_thread[path_thread] > 0) {
                  path2num_thread[path_thread] = path2num_thread[path_thread] - 1.0;
                  std::vector <int> place_vec_thread = doc2place_vec[doc_thread];

                  for (int i = 0; i < int(place_vec_thread.size()); ++i) {
                     int alter_word  = doc_word_topic(place_vec_thread[i], 1);
                     int alter_topic = doc_word_topic(place_vec_thread[i], 4);
                     int hantei_temp = 1;
                     hlda_update_parameters_subtract_thread(alter_topic, alter_word, topic2num_thread, topic2word_map_thread, hantei_temp);
                  } // for(int i=0;i<int(place_vec_thread.size());++i){

                  std::map <int, std::vector <int> > new_path_dict = path_dict;
                  std::map <int, int>         used_path_id, used_topic_id;
                  std::map <std::string, int> path_temp;
                  int num_path_thread = 0;
                  int num_topics_thread = 0;
                  int num_new_path_thread = 0;

                  update_num_path_num_topics_etc(new_path_dict, used_path_id, used_topic_id, path_temp,
                                                 num_path_thread, num_topics_thread, num_new_path_thread);

                  std::vector <int> new_path_id, new_topic_id;
                  for (int i = 0; i < (num_path_thread + num_topics_thread * max_depth_allowed * max_depth_allowed * num_new_path_thread); ++i) {
                     if (used_path_id.find(i) == used_path_id.end()) {
                        new_path_id.push_back(i);
                     }
                  } //for(int i=0;i<num_path+num_new_path;
                  for (int i = 0; i < (num_topics_thread + num_topics_thread * max_depth_allowed * max_depth_allowed * num_new_path_thread); ++i) {
                     if (used_topic_id.find(i) == used_topic_id.end()) {
                        new_topic_id.push_back(i);
                     }
                  } //for(int i=0;i<num_topics+num_new_path;

                  int count_path = 0;
                  int count_topic = 0;
                  for (auto itr = path_temp.begin(); itr != path_temp.end(); ++itr) {
                     std::string temp_string      = itr->first;
                     std::vector <std::string> v1 = Split(temp_string, ',');
                     std::vector <int>         v2;
                     int temp_int = -1;
                     for (int i = 0; i < int(v1.size()); ++i) {
                        if (v1[i] != "new") {
                           temp_int = std::atoi(v1[i].c_str());
                           v2.push_back(temp_int);
                        }else{
                           if (count_topic >= int(new_topic_id.size())) {
                              #pragma omp critical
                              {
                                 stop_hantei = 1;
                                 std::cout << "ERROR NOT ENOUGH ew_topic_id" << std::endl;
                              }
                           }
                           temp_int = new_topic_id[count_topic];  // new topic id
                           v2.push_back(temp_int);
                           count_topic += 1;
                           if (topic2num_thread.find(temp_int) == topic2num_thread.end()) {
                              topic2num_thread[temp_int]      = 0.0;
                              topic2word_map_thread[temp_int] = zero_topic2word_map;
                           }
                           if (topic2num_thread_2.find(temp_int) == topic2num_thread_2.end()) {
                              topic2num_thread_2[temp_int]      = 0.0;
                              topic2word_map_thread_2[temp_int] = zero_topic2word_map;
                           }
                        }
                     } //for(int i = 0;i < int(v1.size());++i){
                     if (count_path >= int(new_path_id.size())) {
                        #pragma omp critical
                        {
                           stop_hantei = 1;
                           std::cout << "ERROR NOT ENOUGH new_path_id" << std::endl;
                        }
                     }
                     temp_int = new_path_id[count_path];
                     new_path_dict[temp_int] = v2;
                     count_path += 1;
                  } //for(auto itr=path_temp.begin();

                  std::map <int, std::vector <int> > extended_path_dict;
                  for (auto itr = new_path_dict.begin(); itr != new_path_dict.end(); ++itr) {
                     std::vector <int> path_vec = itr->second;
                     extended_path_dict[itr->first] = itr->second;
                     int temp_int = -1;
                     if (int(path_vec.size()) < max_depth_allowed) {
                        int going_to_add = max_depth_allowed - int(path_vec.size());
                        for (int l = 0; l < going_to_add; ++l) {
                           if (count_topic >= int(new_topic_id.size())) {
                              #pragma omp critical
                              {
                                 stop_hantei = 1;
                                 std::cout << "ERROR NOT ENOUGH ew_topic_id" << std::endl;
                              }
                           }
                           temp_int = new_topic_id[count_topic];    // new topic id
                           path_vec.push_back(temp_int);
                           count_topic += 1;
                           if (topic2num_thread.find(temp_int) == topic2num_thread.end()) { // used in path sampling
                              topic2num_thread[temp_int]      = 0.0;
                              topic2word_map_thread[temp_int] = zero_topic2word_map;
                           }
                           if (topic2num_thread_2.find(temp_int) == topic2num_thread_2.end()) { // used in level sampling
                              topic2num_thread_2[temp_int]      = 0.0;
                              topic2word_map_thread_2[temp_int] = zero_topic2word_map;
                           }
                           if (count_path >= int(new_path_id.size())) {
                              #pragma omp critical
                              {
                                 stop_hantei = 1;
                                 std::cout << "ERROR NOT ENOUGH new_path_id" << std::endl;
                              }
                           }
                           temp_int = new_path_id[count_path];
                           extended_path_dict[temp_int] = path_vec;
                           count_path += 1;
                        }
                     }
                  }

                  if (thread_id == 0) {
                     new_path_dict_debug      = new_path_dict;
                     extended_path_dict_debug = extended_path_dict;
                  }

                  // Step A: sample path for a document
                  float               agg_topic = 0;
                  float               max_val = -9999999999999;
                  std::vector <int>   path_id_vec;
                  std::vector <float> ratio;
                  for (auto itr = new_path_dict.begin(); itr != new_path_dict.end(); ++itr) {
                     int path_id = itr->first;
                     path_id_vec.push_back(path_id);
                     std::vector <int> topic_vec = itr->second;
                     float             temp      = 0.0;
                     float             temp_in   = 0.0;
                     for (int j = 0; j < int(topic_vec.size()); ++j) {
                        int last_hantei = 0;
                        if (j == int(topic_vec.size()) - 1) {
                           last_hantei = 1;
                        }
                        temp_in = hlda_ncrp_posterior_predictive_robust_rev(doc_thread, topic_vec, j, last_hantei,
                                                                            topic2num_thread, topic2word_map_thread, 1, 0);
                        temp += temp_in;
                     }
                     if (path2num_thread.find(path_id) != path2num_thread.end()) {
                        temp += std::log(path2num_thread[path_id] + path_gamma);
                     }else{
                        temp += std::log(path_gamma);
                     }
                     ratio.push_back(temp);
                     if (max_val < temp) {
                        max_val = temp;
                     }
                  }

                  // sample
                  for (int i = 0; i < int(ratio.size()); ++i) {
                     ratio[i]   = ratio[i] - max_val;
                     ratio[i]   = std::exp(ratio[i]);
                     agg_topic += ratio[i];
                  }
                  double prob[int(ratio.size())];  // Probability array
                  for (int i = 0; i < int(ratio.size()); ++i) {
                     prob[i] = double(ratio[i] / agg_topic);
                  }
                  unsigned int mult_op[ratio.size()];
                  int          num_sample = 1;
                  gsl_ran_multinomial(r[thread_id], int(ratio.size()), num_sample, prob, mult_op);
                  int new_path = -1;
                  for (int i = 0; i < int(ratio.size()); ++i) {
                     if (mult_op[i] == 1) {
                        new_path = path_id_vec[i];
                        break;
                     }
                  }

                  if (new_path < 0) {
                     #pragma omp critical
                     {
                        stop_hantei = 1;
                        std::cout << "ERROR STOP PATH" << std::endl;
                     }
                  }

                  // new_path
                  std::vector <int> new_path_vec;

                  // Step B: sample level
                  if (stop_hantei != 1) {
                     new_path_vec = new_path_dict[new_path];
                     // Create place_vec_thread shuffled
                     std::map <int, std::vector <int> > new_place_path_level;
                     std::random_device seed_gen;
                     std::mt19937       engine(seed_gen());
                     std::shuffle(place_vec_thread.begin(), place_vec_thread.end(), engine);
                     int inner_count = 0;

                     // for each word position
                     for (int itr_place = 0; itr_place < int(place_vec_thread.size()); ++itr_place) {
                        int place     = place_vec_thread[itr_place];
                        int word      = doc_word_topic(place, 1);
                        int old_level = doc_word_topic(place, 3);
                        int old_topic = doc_word_topic(place, 4);

                        int hantei_update_topic2word_map = 1;
                        hlda_update_parameters_subtract_thread(old_topic, word, topic2num_thread_2, topic2word_map_thread_2, hantei_update_topic2word_map);

                        int hantei_update_doc2level = 0;
                        if (doc2level_vec_thread[doc_thread](old_level, 0) > level_gamma) {
                           doc2level_vec_thread[doc_thread](old_level, 0) = doc2level_vec_thread[doc_thread](old_level, 0) - 1.0;
                           hantei_update_doc2level = 1;
                        }

                        int max_depth_thread = 0;
                        for (int i = 0; i < max_depth_allowed; ++i) {
                           if (doc2level_vec_thread[doc_thread](i, 0) > level_gamma) {
                              max_depth_thread = i + 1;
                           }
                        }

                        if (max_depth_thread > max_depth_allowed) {
                           max_depth_thread = max_depth_allowed;
                        }

                        // Create level topic ratio
                        std::vector <float> level_ratio;
                        float agg_level = 0.0;
                        if (level_allocation_type == 0) {   // Pure LDA
                           for (int i = 0; i < int(new_path_vec.size()); ++i) {
                              float temp = alpha_level_vec(i) + doc2level_vec_thread[doc_thread](i, 0);
                              level_ratio.push_back(temp);
                              agg_level += temp;
                           }
                        }else{   // HLDA
                           for (int i = 0; i < max_depth_thread; ++i) {
                              float first       = 1.0;
                              float first_nume  = hyper_m * hyper_pi + doc2level_vec_thread[doc_thread](i, 0);
                              float first_denom = hyper_pi;
                              for (int j = i; j < max_depth_thread; ++j) {
                                 first_denom += doc2level_vec_thread[doc_thread](j, 0);
                              }
                              first = first_nume / first_denom;
                              float second = 1.0;
                              for (int j = 0; j < i; ++j) {
                                 float second_nume_inside = 0.0;
                                 for (int k = j + 1; k < max_depth_thread; ++k) {
                                    second_nume_inside += doc2level_vec_thread[doc_thread](k, 0);
                                 }
                                 float second_nume         = (1.0 - hyper_m) * hyper_pi + second_nume_inside;
                                 float second_denom_inside = 0.0;
                                 for (int k = j; k < max_depth_thread; ++k) {
                                    second_denom_inside += doc2level_vec_thread[doc_thread](k, 0);
                                 }
                                 float second_denom = hyper_pi + second_denom_inside;
                                 second = second * (second_nume / second_denom);
                              }
                              float temp = first * second;
                              level_ratio.push_back(temp);
                              agg_level += temp;
                           }
                           if (max_depth_thread != max_depth_allowed) {
                              // ADD another level
                              float temp_2 = 1 - agg_level;
                              if (temp_2 < 0.001) {
                                 temp_2 = 0.001;
                              }
                              level_ratio.push_back(temp_2);
                              if (int(level_ratio.size()) < int(new_path_vec.size())) {
                                 level_ratio.push_back(0.001);
                              }
                           }
                        }   //if(level_allocation_type == 0){}else{}

                        float agg_topic = 0;
                        max_val = -9999999999999;
                        std::vector <float> ratio, ratio_a, ratio_b;
                        std::vector <int>   topic_temp;
                        for (int j = 0; j < int(level_ratio.size()); ++j) {
                           int topic = -1;
                           if (j < int(new_path_vec.size())) {
                              topic = new_path_vec[j];
                           }
                           topic_temp.push_back(topic);
                           float log_prob_a = std::log(level_ratio[j]);
                           float log_prob_b = hlda_posterior_predictive_single_robust_rev(place, j, topic,
                                                                                          topic2num_thread_2, topic2word_map_thread_2, 1, 0);
                           float log_prob = log_prob_a + log_prob_b;
                           ratio_a.push_back(log_prob_a);
                           ratio_b.push_back(log_prob_b);
                           ratio.push_back(log_prob);
                           if (max_val < ratio[j]) {
                              max_val = ratio[j];
                           }
                        }
                        agg_topic = 0;
                        for (int j = 0; j < int(ratio.size()); ++j) {
                           ratio[j]   = ratio[j] - max_val;
                           ratio[j]   = std::exp(ratio[j]);
                           agg_topic += ratio[j];
                        }

                        // sample
                        double prob[ratio.size()];    // Probability array
                        for (int j = 0; j < int(ratio.size()); ++j) {
                           prob[j] = double(ratio[j] / agg_topic);
                        }
                        unsigned int mult_op[int(ratio.size())];
                        int          num_sample = 1;
                        gsl_ran_multinomial(r[thread_id], int(ratio.size()), num_sample, prob, mult_op);
                        int new_level = -1;
                        int new_topic = -1;
                        for (int j = 0; j < int(ratio.size()); ++j) {
                           if (mult_op[j] == 1) {
                              new_level = j;
                              break;
                           }
                        }

                        if (new_level == -1) {
                           #pragma omp critical
                           {
                              stop_hantei = 1;
                              std::cout << "ERROR STOP LEVEL" << std::endl;
                           }

                           if (verbose == 3) {
                              std::cout << "the new_level == -1!!!!!!!!!!!!" << std::endl;
                              for (int p = 0; p < int(ratio.size()); ++p) {
                                 std::cout << "level: " << p << " ratio_a: " << ratio_a[p] << ", ratio_b: " << ratio_b[p] << ", topic2num_thread_2[topic]: " << topic2num_thread_2[topic_temp[p]] << std::endl;
                              }
                           }
                        }

                        if (new_level < int(new_path_vec.size())) {
                           new_topic = new_path_vec[new_level];
                        }else{
                           for (auto itr = extended_path_dict.begin(); itr != extended_path_dict.end(); ++itr) {
                              std::vector <int> path_vec = itr->second;
                              if (int(path_vec.size()) == (new_level + 1)) {
                                 int hantei_replace = 1;
                                 for (int k = 0; k < int(new_path_vec.size()); ++k) {
                                    if (new_path_vec[k] != path_vec[k]) {
                                       hantei_replace = 0;
                                    }
                                 }
                                 if (hantei_replace == 1) {
                                    new_path     = itr->first;
                                    new_path_vec = path_vec;
                                    new_topic    = path_vec[new_level];
                                    break;
                                 }
                              }
                           }
                        }

                        if (new_topic < 0) {
                           #pragma omp critical
                           {
                              stop_hantei = 1;
                              std::cout << "ERROR STOP LEVEL TOPIC" << std::endl;
                           }
                        }

                        if (hantei_update_topic2word_map == 1) {
                           hlda_update_parameters_add_thread(new_topic, word, topic2num_thread_2, topic2word_map_thread_2);
                        }

                        if (hantei_update_doc2level == 1) {
                           float temp_val = doc2level_vec_thread[doc_thread](new_level, 0) + 1.0;
                           doc2level_vec_thread[doc_thread](new_level, 0) = temp_val;
                        }
                        std::vector <int> result;
                        result.push_back(place);
                        result.push_back(new_path);
                        result.push_back(new_level);
                        result.push_back(new_topic);
                        new_place_path_level[inner_count] = result;
                        inner_count += 1;
                     }  //for (int itr_place = 0; itr_place < int(place_vec_thread.size()); ++itr_place) {

                     #pragma omp critical
                     {
                        //if(discard_flag == 0){
                        if (1 == 1) {
                           sample_counter += 1.0;
                           int hantei     = 1;
                           int change_doc = -1;

                           for (auto itr2 = new_place_path_level.begin(); itr2 != new_place_path_level.end(); ++itr2) {
                              std::vector <int> result = itr2->second;
                              int change_place         = result[0];
                              change_doc = doc_word_topic(change_place, 0);
                              //int new_path  = result[1];
                              int new_level = result[2];
                              int new_topic = result[3];
                              int old_path  = doc_word_topic(change_place, 2);
                              int old_level = doc_word_topic(change_place, 3);
                              int old_topic = doc_word_topic(change_place, 4);

                              //if (new_level > -1 && new_topic > -1 && new_path > -1){
                              if (1 == 1) {
                                 doc_word_topic(change_place, 2) = new_path;
                                 doc_word_topic(change_place, 3) = new_level;
                                 doc_word_topic(change_place, 4) = new_topic;
                                 if (doc2level_vec[change_doc](old_level, 0) > level_gamma) {
                                    doc2level_vec[change_doc](old_level, 0) = doc2level_vec[change_doc](old_level, 0) - 1.0;
                                    doc2level_vec[change_doc](new_level, 0) = doc2level_vec[change_doc](new_level, 0) + 1.0;
                                 }

                                 if (1 == 1) {
                                    if (hantei == 1) {   // we only need to do this once
                                       hantei = 0;
                                       if (path_dict.find(old_path) != path_dict.end()) {
                                          path2num[old_path] = path2num[old_path] - 1.0;
                                          if (path2num[old_path] < 0.1) {// recrate path2num and path_dict
                                             std::map <int, std::vector <int> > path_dict_t;
                                             std::unordered_map <int, float>    path2num_t;
                                             for (auto itr4 = path2num.begin(); itr4 != path2num.end(); ++itr4) {
                                                int path_num_t = itr4->first;
                                                if (path2num[path_num_t] > 0.1) {
                                                   path2num_t[path_num_t]  = itr4->second;
                                                   path_dict_t[path_num_t] = path_dict[path_num_t];
                                                }
                                             }
                                             path2num.clear();
                                             path2num = path2num_t;
                                             path_dict.clear();
                                             path_dict = path_dict_t;
                                             // OLD VERSION
                                             //path_dict.erase(old_path);
                                             //path2num.erase(old_path);
                                          }
                                       }
                                       if (path_dict.find(new_path) != path_dict.end()) {
                                          path2num[new_path] = path2num[new_path] + 1.0;
                                       }else{   // Completely new path
                                          path_dict[new_path] = new_path_vec;
                                          path2num[new_path]  = 1.0;
                                          for (int k = 0; k < int(new_path_vec.size()); ++k) {
                                             if (topic2num.find(new_path_vec[k]) == topic2num.end()) {
                                                topic2num[new_path_vec[k]] = 0.0;
                                             }
                                             if (topic2word_map.find(new_path_vec[k]) == topic2word_map.end()) {
                                                topic2word_map[new_path_vec[k]] = zero_topic2word_map;
                                             }
                                          }
                                       }
                                       doc2path[change_doc] = new_path;
                                       update_num_path_num_topics(path_dict);
                                    }   //if(hantei==1){
                                    int change_word = doc_word_topic(change_place, 1);
                                    hlda_update_parameters_subtract(old_topic, change_word);
                                    hlda_update_parameters_add(new_topic, change_word);
                                 } //if (1 == 1){}
                              }    //if (new_level > -1 && new_topic > -1 && new_path > -1){
                           }       //for(auto itr2=new_place_path_level.begin()
                        }          //if(discard_flag == 0){
                     }             //#pragma omp critical
                  } //if (stop_hantei != 1) {
               } //if (1 == 1) {
               if (verbose == 1) {
                  std::cout << "1 == 1" << std::endl;
               }
            }    //if (stop_hantei != 1) {
            if (verbose == 1) {
               std::cout << "itr_inner " << itr_inner << std::endl;
            }
         }   //for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
         if (verbose == 1) {
            std::cout << "itr_outer " << itr_outer << std::endl;
         }
      }      //for (int itr_outer = 0; itr_outer < num_iteration; ++itr_outer) {

      if (verbose == 1) {
         std::cout << "Before Free memory" << std::endl;
      }

      // free memory
      for (int i = 0; i < num_threads; i++) {
         gsl_rng_free(r[i]);
      }

      if (verbose == 1) {
         std::cout << "After Free memory" << std::endl;
      }
   }//hlda_ncrp_collapsed_gibbs_sample_parallel

   float hlda_ncrp_posterior_predictive_robust_rev_test(const Eigen::MatrixXi& doc_word_topic_thread_temp,
                                                        const std::vector <int>& topic_vec,
                                                        const int& level,
                                                        const int& last_topic,
                                                        std::unordered_map <int, float>& topic2num_temp,
                                                        std::unordered_map <int, std::unordered_map <int, float> >& topic2word_map_temp,
                                                        const int& log_true, const int& verbose) {
      std::vector <int> place_level_vec;
      for (int i = 0; i < int(doc_word_topic_thread_temp.rows()); ++i) {
         if (last_topic != 1) {
            if (doc_word_topic_thread_temp(i, 3) == level) {
               place_level_vec.push_back(i);
            }
         }else{
            if (doc_word_topic_thread_temp(i, 3) >= level) {
               place_level_vec.push_back(i);
            }
         }
      }

      float     eta_in_hLDA = level2eta_in_HLDA[level];
      const int topic       = topic_vec[level];
      float     n           = float(topic2num_temp[topic]);
      float     m           = float(place_level_vec.size());
      float     log_prob    = 0.0;
      float     prob        = 0.0;

      float second_nume_first_denom = 0.0;
      std::unordered_map <int, float> word_map_temp = topic2word_map_temp[topic];
      for (auto itr = word_map_temp.begin(); itr != word_map_temp.end(); ++itr) {
         int   word_0 = itr->first;
         float temp   = itr->second;
         //first_denom = first_denom + std::lgamma(temp + eta_in_hLDA);
         float temp2 = 0;
         for (int i = 0; i < int(place_level_vec.size()); ++i) {
            if (doc_word_topic_thread_temp(place_level_vec[i], 1) == word_0) {
               temp2 = temp2 + 1.0;
            }
         }
         //second_nume = second_nume + std::lgamma(temp2 + temp + eta_in_hLDA);
         for (int jj = 0; jj < int(temp2); ++jj) {
            second_nume_first_denom += std::log(temp + eta_in_hLDA + float(jj));
         }
      }

      //first_nume = std::lgamma(n + voc * eta_in_hLDA);
      //second_denom = std::lgamma(n + m + voc * eta_in_hLDA);
      float first_nume_second_denom = 0.0;
      float second_denom_first_nume = 0.0;
      for (int jj = 0; jj < int(m); ++jj) {
         second_denom_first_nume += std::log(n + voc * eta_in_hLDA + float(jj));
      }
      first_nume_second_denom = -second_denom_first_nume;

      //log_prob = first_nume - second_denom - first_denom + second_nume ;
      log_prob = first_nume_second_denom + second_nume_first_denom;

      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      return(prob);
   }

   float hlda_posterior_predictive_single_robust_rev_test(const Eigen::MatrixXi& doc_word_topic_thread_temp,
                                                          const int& place,
                                                          const int& level,
                                                          const int& topic,
                                                          std::unordered_map <int, float>& topic2num_temp,
                                                          std::unordered_map <int, std::unordered_map <int, float> >& topic2word_map_temp,
                                                          const int& log_true, const int& verbose) {
      float eta_in_hLDA = level2eta_in_HLDA[level];

      //std::vector<int> place_vec = doc2place_vec[doc];
      std::vector <int> place_level_vec;
      place_level_vec.push_back(place);

      // KOKOGA MONDAI
      float n        = float(topic2num_temp[topic]);
      float m        = 1;
      float log_prob = 0.0;
      float prob     = 0.0;

      //float first_denom_second_nume = 0.0;
      float second_nume_first_denom = 0.0;
      //if (topic2word_map_temp.find(topic) != topic2word_map_temp.end()) {
      std::unordered_map <int, float> word_map_temp = topic2word_map_temp[topic];
      for (auto itr = word_map_temp.begin(); itr != word_map_temp.end(); ++itr) {
         int   word_0 = itr->first;
         float temp   = itr->second;
         //first_denom = first_denom + std::lgamma(temp + eta_in_hLDA);
         float temp2 = 0;
         for (int i = 0; i < int(place_level_vec.size()); ++i) {
            if (doc_word_topic_thread_temp(place_level_vec[i], 1) == word_0) {
               temp2 = temp2 + 1.0;
            }
         }
         //second_nume = second_nume + std::lgamma(temp2 + temp + eta_in_hLDA);
         for (int jj = 0; jj < int(temp2); ++jj) {
            second_nume_first_denom += std::log(temp + eta_in_hLDA + float(jj));
         }
      }

      //first_nume = std::lgamma(n + voc * eta_in_hLDA);
      //second_denom = std::lgamma(n + m + voc * eta_in_hLDA);
      float first_nume_second_denom = 0.0;
      float second_denom_first_nume = 0.0;
      for (int jj = 0; jj < m; ++jj) {
         second_denom_first_nume += std::log(n + voc * eta_in_hLDA + float(jj));
      }
      first_nume_second_denom = -second_denom_first_nume;

      //log_prob = first_nume - second_denom - first_denom + second_nume ;
      log_prob = first_nume_second_denom + second_nume_first_denom;

      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }

      return(prob);
   }

   void hlda_ncrp_evaluate_held_out_log_likelihood(const std::vector <int> doc_test_ids, const int num_iteration, const int burn_in, const float path_thres, const int approx, int num_threads, int verbose) {

      if (verbose == 1) {
         std::cout << "Enter lda_evaluate_held_out_log_likelihood" << std::endl;
      }

      if (num_threads < 1) {
         num_threads = omp_get_max_threads();
         std::cout << "Using " << num_threads << " threads" << std::endl;
      }

      omp_set_num_threads(num_threads);
      int max_num_threads = omp_get_max_threads();
      const gsl_rng_type *T;
      gsl_rng **          r;
      gsl_rng_env_setup();
      T = gsl_rng_mt19937;
      r = (gsl_rng **)malloc(max_num_threads * sizeof(gsl_rng *));
      std::random_device rand_device;
      std::mt19937       mt(rand_device());
      for (int i = 0; i < max_num_threads; i++) {
         r[i] = gsl_rng_alloc(T);
         gsl_rng_set(r[i], mt());
      }

      if (verbose == 1) {
         std::cout << "Before pragma omp parallel for" << std::endl;
      }

      float counter = 0.0;
      #pragma omp parallel for
      for (int k = 0; k < int(doc_test_ids.size()); ++k) {
         std::unordered_map <int, std::unordered_map <int, float> > topic2word_map_thread = topic2word_map;
         std::unordered_map <int, int> update_topic_thread;
         int             doc_test = doc_test_ids[k];
         int             doc_size = int(doc2place_vec_test[doc_test].size());
         Eigen::MatrixXi doc_word_topic_thread = Eigen::MatrixXi::Zero(doc_size, 8);  // take log sum of this
         for (int i = 0; i < doc_size; ++i) {
            int place_t = doc2place_vec_test[doc_test][i];
            doc_word_topic_thread(i, 0) = doc_word_topic_test(place_t, 0);
            doc_word_topic_thread(i, 1) = doc_word_topic_test(place_t, 1);
            doc_word_topic_thread(i, 2) = doc_word_topic_test(place_t, 2);
            doc_word_topic_thread(i, 3) = doc_word_topic_test(place_t, 3);
         }

         if (verbose == 1) {
            std::cout << "Before for (int up_to = 0" << std::endl;
         }

         // for every word position
         Eigen::MatrixXf heldout_prob = Eigen::MatrixXf::Zero(doc_word_topic_thread.rows(), 1);
         std::unordered_map <int, std::unordered_map <int, float> > place2level_weight_thread;
         for (int up_to = 0; up_to < int(doc_word_topic_thread.rows()); ++up_to) {
            int thread_id = omp_get_thread_num();
            std::unordered_map <int, float> topic2num_thread = topic2num;
            std::unordered_map <int, float> path2num_thread  = path2num;
            Eigen::MatrixXf level_vec_sub = Eigen::MatrixXf::Zero(num_topics, 1);
            Eigen::MatrixXi doc_word_topic_thread_temp = Eigen::MatrixXi::Zero(up_to + 1, 10);

            for (int i = 0; i < (up_to + 1); ++i) {
               doc_word_topic_thread_temp(i, 0) = doc_word_topic_thread(i, 0);
               doc_word_topic_thread_temp(i, 1) = doc_word_topic_thread(i, 1);
               doc_word_topic_thread_temp(i, 2) = doc_word_topic_thread(i, 2);
               doc_word_topic_thread_temp(i, 3) = doc_word_topic_thread(i, 3);
               int level = doc_word_topic_thread_temp(i, 3);
               if (i == up_to) {
                  level_vec_sub(level, 0) += 1.0;
               }else{
                  for (auto itr3 = place2level_weight_thread[i].begin(); itr3 != place2level_weight_thread[i].end(); ++itr3) {
                     level = itr3->first;
                     float val = itr3->second;
                     level_vec_sub(level, 0) += val;
                  } // for (auto itr3 = place2topic_weight_thread[i].begin()
               } // if (i == up_to) {
            }    //for (int i = 0; i < (up_to + 1); ++i) {

            float agg   = 0;
            float count = 0;
            update_topic_thread.clear();
            std::unordered_map <int, float> level_weight_thread;
            for (int rep = 0; rep < num_iteration; ++rep) {
               // Step A: sample path for a document
               float                           agg_topic = 0;
               float                           max_val   = -9999999999999;
               std::vector <int>               path_id_vec;
               std::vector <float>             ratio;
               std::vector <std::string>       topic_string_vec;
               std::unordered_map <int, float> topic2third_nume_inside;

               std::unordered_map <int, float> topic2temp_in;
               for (auto itr = path_dict.begin(); itr != path_dict.end(); ++itr) {
                  int path_id = itr->first;
                  path_id_vec.push_back(path_id);
                  std::vector <int> topic_vec = itr->second;
                  float             temp      = 0.0;
                  float             temp_in   = 0.0;

                  for (int j = 0; j < int(topic_vec.size()); ++j) {
                     int topic_temp = topic_vec[j];
                     int last_hantei = 0;
                     if (j == int(topic_vec.size()) - 1) {
                        last_hantei = 1;
                     }
                     if(topic2temp_in.find(topic_temp)==topic2temp_in.end()){
                         temp_in = hlda_ncrp_posterior_predictive_robust_rev_test(doc_word_topic_thread_temp,
                                                                                  topic_vec, j, last_hantei, topic2num_thread, topic2word_map_thread, 1, 0);
                         topic2temp_in[topic_temp] = temp_in;
                     }else{
                         temp_in = topic2temp_in[topic_temp];
                     }
                     temp += temp_in;
                  }
                  if (path2num_thread.find(path_id) != path2num_thread.end()) {
                     temp += std::log(path2num_thread[path_id] + path_gamma);
                  }else{
                     temp += std::log(path_gamma);
                  }
                  ratio.push_back(temp);
                  if (max_val < temp) {
                     max_val = temp;
                  }
               }//for (auto itr = path_dict.begin(); itr != path_dict.end(); ++itr) {

               // sample
               for (int i = 0; i < int(ratio.size()); ++i) {
                  ratio[i]   = ratio[i] - max_val;
                  ratio[i]   = std::exp(ratio[i]);
                  agg_topic += ratio[i];
               }
               double prob_path[int(ratio.size())];   // Probability array
               for (int i = 0; i < int(ratio.size()); ++i) {
                  prob_path[i] = double(ratio[i] / agg_topic);
               }

               unsigned int mult_op[ratio.size()];
               int          num_sample = 1;
               gsl_ran_multinomial(r[thread_id], int(ratio.size()), num_sample, prob_path, mult_op);
               int new_path = -1;
               std::vector<int> path_id_vec_2;
               std::vector<float> prob_path_2;
               for (int i = 0; i < int(ratio.size()); ++i) {
                  if(prob_path[i] > path_thres){
                      prob_path_2.push_back(float(prob_path[i]));
                      path_id_vec_2.push_back(path_id_vec[i]);
                  }
                  if (mult_op[i] == 1) {
                     new_path = path_id_vec[i];
                  }
               }

               for (int kkk = 0; kkk < int(prob_path_2.size()); ++kkk) {
                   new_path = path_id_vec_2[kkk];
                   std::vector <int> new_path_vec;
                   new_path_vec = path_dict[new_path];
                   update_topic_thread.clear();
                   std::unordered_map <int, float> topic_weight_thread;
                   int place = int(doc_word_topic_thread_temp.rows()) - 1;
                   int word      = doc_word_topic_thread_temp(place, 1);
                   int old_level = doc_word_topic_thread_temp(place, 3);

                   // Create level topic ratio
                   std::vector <float> level_ratio;
                   float agg_level        = 0.0;
                   int   max_depth_thread = max_depth_allowed;
                   if (level_allocation_type == 0) {    // Pure LDA
                      for (int i = 0; i < int(new_path_vec.size()); ++i) {
                         float temp = alpha_level_vec(i) + level_vec_sub(i, 0);
                         level_ratio.push_back(temp);
                         agg_level += temp;
                      }
                   }else{    // HLDA
                      for (int i = 0; i < max_depth_thread; ++i) {
                         float first       = 1.0;
                         float first_nume  = hyper_m * hyper_pi + level_vec_sub(i, 0);
                         float first_denom = hyper_pi;
                         for (int j = i; j < max_depth_thread; ++j) {
                            first_denom += level_vec_sub(j, 0);
                         }
                         first = first_nume / first_denom;
                         float second = 1.0;
                         for (int j = 0; j < i; ++j) {
                            float second_nume_inside = 0.0;
                            for (int k = j + 1; k < max_depth_thread; ++k) {
                               second_nume_inside += level_vec_sub(k, 0);
                            }
                            float second_nume         = (1.0 - hyper_m) * hyper_pi + second_nume_inside;
                            float second_denom_inside = 0.0;
                            for (int k = j; k < max_depth_thread; ++k) {
                               second_denom_inside += level_vec_sub(k, 0);
                            }
                            float second_denom = hyper_pi + second_denom_inside;
                            second = second * (second_nume / second_denom);
                         }
                         float temp = first * second;
                         level_ratio.push_back(temp);
                         agg_level += temp;
                      }
                   }    //if(level_allocation_type == 0){}else{}

                   max_val = -9999999999999;
                   ratio.clear();
                   for (int j = 0; j < int(level_ratio.size()); ++j) {
                      int topic = -1;
                      if (j < int(new_path_vec.size())) {
                         topic = new_path_vec[j];
                      }
                      float log_prob_a = std::log(level_ratio[j]);
                      float log_prob_b = hlda_posterior_predictive_single_robust_rev_test(doc_word_topic_thread_temp, place,
                                                                       j, topic,topic2num_thread, topic2word_map_thread, 1, 0);
                      float log_prob = log_prob_a + log_prob_b;
                      ratio.push_back(log_prob);
                      if (max_val < ratio[j]) {
                         max_val = ratio[j];
                      }
                   } //for (int j = 0; j < int(level_ratio.size()); ++j) {

                   agg_topic = 0;
                   for (int j = 0; j < int(ratio.size()); ++j) {
                      ratio[j]   = ratio[j] - max_val;
                      ratio[j]   = std::exp(ratio[j]);
                      agg_topic += ratio[j];
                   }
                   // sample
                   double prob_level[ratio.size()];     // Probability array
                   for (int j = 0; j < int(ratio.size()); ++j) {
                      prob_level[j] = double(ratio[j] / agg_topic);
                   }
                   unsigned int mult_op_level[int(ratio.size())];
                   num_sample = 1;
                   gsl_ran_multinomial(r[thread_id], int(ratio.size()), num_sample, prob_level, mult_op_level);
                   int new_level = -1;
                   int new_topic = -1;
                   for (int j = 0; j < int(ratio.size()); ++j) {
                      if (mult_op_level[j] == 1) {
                         new_level = j;
                         break;
                      }
                   }
                   level_vec_sub(old_level, 0)         -= 1.0;
                   level_vec_sub(new_level, 0)         += 1.0;
                   doc_word_topic_thread_temp(place, 3) = new_level;
                   doc_word_topic_thread(place, 3)      = new_level;
                   new_topic = path_dict[new_path][new_level];
                   if (rep > burn_in) {
                      update_topic_thread.clear();
                      for (int j = 0; j < int(ratio.size()); ++j) {
                          new_level = j;
                          new_topic = path_dict[new_path][new_level];
                          if(topic2word_map_thread[new_topic].find(word)!=topic2word_map_thread[new_topic].end()){
                            agg += float(prob_path_2[kkk]) * float(prob_level[j]) * (topic2word_map_thread[new_topic][word] + beta) / (voc * beta + topic2num_thread[new_topic]);
                          }else{
                            agg += float(prob_path_2[kkk]) * float(prob_level[j]) * beta / (voc * beta + topic2num_thread[new_topic]);
                          }
                      }
                      count += float(prob_path_2[kkk]);
                      if (level_weight_thread.find(new_level) == level_weight_thread.end()) {
                         level_weight_thread[new_level] = 1.0;
                      }else{
                         level_weight_thread[new_level] += 1.0;
                      }
                   } //if (rep > burn_in) {
                } //for (int kkk = 0; kkk < int(ratio.size()); ++kkk) {
            }    //for (int rep = 0; rep < num_iteration; ++rep) {

            float prob_2 = agg / count;
            heldout_prob(up_to, 0) = prob_2;

            // Update place2topic_weight_thread
            float agg2 = 0.0;
            for (auto itr3 = level_weight_thread.begin(); itr3 != level_weight_thread.end(); ++itr3) {
               agg2 += 1.0;
            }
            for (auto itr3 = level_weight_thread.begin(); itr3 != level_weight_thread.end(); ++itr3) {
               itr3->second = itr3->second / agg2;
            }
            place2level_weight_thread[up_to] = level_weight_thread;
         }//for (int up_to = 0; up_to < int(doc_word_topic_thread.rows()); ++up_to) {
         #pragma omp critical
         {
            doc_test2heldout_prob[doc_test] = heldout_prob;
            counter += 1;
            std::cout << "Finished: " << counter / float(doc_test_ids.size()) << std::endl;
         } //#pragma omp critical
      }    //for (int k = 0; k < int(doc_test_ids.size()); ++k)
   }

   // HLDA-ncrp: END

   // GLDA: START //
   Eigen::VectorXf alpha_topic_vec;
   std::unordered_map <long long, Eigen::MatrixXf> doc2topic_vec, doc2topic_vec_test;
   Eigen::MatrixXf mu_0, Sigma_0;
   float num_0 = 0.0;

   void calc_variance() {
      mu_0    = zero_mu;
      Sigma_0 = zero_Sigma;
      for (int i = 0; i < doc_word_topic.rows(); ++i) {
         int             word = doc_word_topic(i, 1);
         Eigen::MatrixXf x    = embedding_matrix.row(word);
         num_0 += 1;
         mu_0  += x;
      }
      mu_0 = mu_0 / num_0;
      for (int i = 0; i < doc_word_topic.rows(); ++i) {
         int             word         = doc_word_topic(i, 1);
         Eigen::MatrixXf x            = embedding_matrix.row(word);
         Eigen::MatrixXf outer_matrix = (x - mu_0.transpose()) * (x - mu_0);
         Sigma_0 += outer_matrix;
      }
      Sigma_0 = Sigma_0 / (num_0 - 1.0);
   }

   void create_doc2place_vec() {
      doc2place_vec.clear();
      num_docs = 0;
      for (int i = 0; i < doc_word_topic.rows(); ++i) {
         //int doc = doc_word_topic(i, 0);
         if (doc2place_vec.find(doc_word_topic(i, 0)) == doc2place_vec.end()) {
            std::vector <int> temp_vec;
            temp_vec.push_back(i);
            doc2place_vec[doc_word_topic(i, 0)] = temp_vec;
            num_docs += 1;
         }else{
            doc2place_vec[doc_word_topic(i, 0)].push_back(i);
         }
      }
   }

   void glda_calc_tables_parameters_from_assign() {
      if (num_topics == 0 || embedding_dimension == 0) {
         std::cout << "Please set num_topics and embedding_dimension" << std::endl;
      }else{
         topic2num.clear();
         topic2mu.clear();
         topic2Sigma.clear();
         doc2topic_vec.clear();
         for (int i = 0; i < doc_word_topic.rows(); ++i) {
            int             doc   = doc_word_topic(i, 0);
            int             word  = doc_word_topic(i, 1);
            int             topic = doc_word_topic(i, 2);
            Eigen::MatrixXf x     = embedding_matrix.row(word);
            // Create tables
            if (doc2topic_vec.find(doc) == doc2topic_vec.end()) {
               doc2topic_vec[doc] = Eigen::MatrixXf::Zero(num_topics, 1);
            }
            doc2topic_vec[doc](topic, 0) += 1.0;

            // Create Parameters
            if (topic2num.find(topic) == topic2num.end()) {
               topic2num[topic] = 1.0;
            }else{
               topic2num[topic] += 1.0;
            }
            if (topic2mu.find(topic) == topic2mu.end()) {
               topic2mu[topic] = x;
            }else{
               topic2mu[topic] += x;
            }
         }
         // Normalize
         for (int i = 0; i < num_topics; ++i) {
            if (topic2num.find(i) != topic2num.end()) {
               topic2mu[i] = topic2mu[i] / topic2num[i];
            }
         }
         // Variance
         for (int i = 0; i < doc_word_topic.rows(); ++i) {
            int             word  = doc_word_topic(i, 1);
            int             topic = doc_word_topic(i, 2);
            Eigen::MatrixXf x     = embedding_matrix.row(word);
            if (topic2Sigma.find(topic) == topic2Sigma.end()) {
               Eigen::MatrixXf outer_matrix = (x - topic2mu[topic]).transpose() * (x - topic2mu[topic]);
               topic2Sigma[topic] = outer_matrix;
            }else{
               Eigen::MatrixXf outer_matrix = (x - topic2mu[topic]).transpose() * (x - topic2mu[topic]);
               topic2Sigma[topic] += outer_matrix;
            }
         }
      }
   }

   void glda_update_parameters_subtract(const int topic, const int wordid) {
      // word embedding : x (1,embedding_dimension)
      Eigen::MatrixXf x  = embedding_matrix.row(wordid);
      float           N0 = topic2num[topic];

      if (N0 < 1.1) {
         topic2mu[topic]    = zero_mu;
         topic2Sigma[topic] = zero_Sigma;
         topic2num[topic]   = 0.0;
      }else{
         Eigen::MatrixXf mu0       = topic2mu[topic];
         Eigen::MatrixXf Sigma0    = topic2Sigma[topic];
         Eigen::MatrixXf mu0_minus = (N0 * mu0 - x) / (N0 - 1.0);
         // (4*Sigma2 + 4*(m1-m2).transpose()*(m1-m2) - (x4-m1).transpose()*(x4-m1)) /3
         Eigen::MatrixXf Sigma0_minus = Sigma0 + N0 * (mu0_minus - mu0).transpose() * (mu0_minus - mu0) - (x - mu0_minus).transpose() * (x - mu0_minus);
         topic2mu[topic]    = mu0_minus;
         topic2Sigma[topic] = Sigma0_minus;
         topic2num[topic]   = N0 - 1.0;
      }
   }

   void glda_update_parameters_add(const int& new_topic, const int wordid) {
      // Update parameters
      float N3 = topic2num[new_topic];

      // word embedding : x (1,embedding_dimension)
      Eigen::MatrixXf x        = embedding_matrix.row(wordid);
      Eigen::MatrixXf mu3      = topic2mu[new_topic];
      Eigen::MatrixXf Sigma3   = topic2Sigma[new_topic];
      Eigen::MatrixXf mu3_plus = (N3 * mu3 + x) / (N3 + 1.0);
      // # (3*Sigma1 + 3*(m1-m2).transpose()*(m1-m2) + (x4-m2).transpose()*(x4-m2)) / 4
      Eigen::MatrixXf Sigma3_plus = Sigma3 + N3 * (mu3 - mu3_plus).transpose() * (mu3 - mu3_plus) + (x - mu3_plus).transpose() * (x - mu3_plus);
      topic2num[new_topic]   = N3 + 1.0;
      topic2mu[new_topic]    = mu3_plus;
      topic2Sigma[new_topic] = Sigma3_plus;
   }

   void glda_update_parameters_subtract_thread(const int topic, const int wordid,
                                               std::unordered_map <int, float>& topic2num_thread, std::unordered_map <int, Eigen::MatrixXf>& topic2mu_thread,
                                               std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_thread) {
      // word embedding : x (1,embedding_dimension)
      Eigen::MatrixXf x  = embedding_matrix.row(wordid);
      float           N0 = topic2num_thread[topic];

      if (N0 < 1.1) {
         topic2mu_thread[topic]    = zero_mu;
         topic2Sigma_thread[topic] = zero_Sigma;
         topic2num_thread[topic]   = 0.0;
      }else{
         Eigen::MatrixXf mu0       = topic2mu_thread[topic];
         Eigen::MatrixXf Sigma0    = topic2Sigma_thread[topic];
         Eigen::MatrixXf mu0_minus = (N0 * mu0 - x) / (N0 - 1.0);
         // (4*Sigma2 + 4*(m1-m2).transpose()*(m1-m2) - (x4-m1).transpose()*(x4-m1)) /3
         Eigen::MatrixXf Sigma0_minus = Sigma0 + N0 * (mu0_minus - mu0).transpose() * (mu0_minus - mu0) - (x - mu0_minus).transpose() * (x - mu0_minus);
         topic2mu_thread[topic]    = mu0_minus;
         topic2Sigma_thread[topic] = Sigma0_minus;
         topic2num_thread[topic]   = N0 - 1.0;
      }
   }

   void glda_update_parameters_add_thread(const int& new_topic, const int wordid,
                                          std::unordered_map <int, float>& topic2num_thread, std::unordered_map <int, Eigen::MatrixXf>& topic2mu_thread,
                                          std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_thread
                                          ) {
      // Update parameters
      float N3 = topic2num_thread[new_topic];

      // word embedding : x (1,embedding_dimension)
      Eigen::MatrixXf x        = embedding_matrix.row(wordid);
      Eigen::MatrixXf mu3      = topic2mu_thread[new_topic];
      Eigen::MatrixXf Sigma3   = topic2Sigma_thread[new_topic];
      Eigen::MatrixXf mu3_plus = (N3 * mu3 + x) / (N3 + 1.0);
      // # (3*Sigma1 + 3*(m1-m2).transpose()*(m1-m2) + (x4-m2).transpose()*(x4-m2)) / 4
      Eigen::MatrixXf Sigma3_plus = Sigma3 + N3 * (mu3 - mu3_plus).transpose() * (mu3 - mu3_plus) + (x - mu3_plus).transpose() * (x - mu3_plus);
      topic2num_thread[new_topic]   = N3 + 1.0;
      topic2mu_thread[new_topic]    = mu3_plus;
      topic2Sigma_thread[new_topic] = Sigma3_plus;
   }

   void glda_update_parameters_add_thread_weighted(const int& new_topic, const int wordid,
                                                   std::unordered_map <int, float>& topic2num_thread, std::unordered_map <int, Eigen::MatrixXf>& topic2mu_thread,
                                                   std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_thread, const float& weight
                                                   ) {
      // Update parameters
      float N3 = topic2num_thread[new_topic];

      // word embedding : x (1,embedding_dimension)
      Eigen::MatrixXf x        = embedding_matrix.row(wordid);
      Eigen::MatrixXf mu3      = topic2mu_thread[new_topic];
      Eigen::MatrixXf Sigma3   = topic2Sigma_thread[new_topic];
      Eigen::MatrixXf mu3_plus = (N3 * mu3 + x) / (N3 + weight);
      // # (3*Sigma1 + 3*(m1-m2).transpose()*(m1-m2) + (x4-m2).transpose()*(x4-m2)) / 4
      Eigen::MatrixXf Sigma3_plus = Sigma3 + N3 * (mu3 - mu3_plus).transpose() * (mu3 - mu3_plus) + (x - mu3_plus).transpose() * (x - mu3_plus);
      topic2num_thread[new_topic]   = N3 + weight;
      topic2mu_thread[new_topic]    = mu3_plus;
      topic2Sigma_thread[new_topic] = Sigma3_plus;
   }

   void ghlda_update_mu_sigma(const int& new_topic, const Eigen::MatrixXf& x,
                              float& N_temp,
                              Eigen::MatrixXf& mu_temp, Eigen::MatrixXf& Sigma_temp) {
      // Update parameters
      float N3 = N_temp;

      N_temp = N3 + 1.0;
      Eigen::MatrixXf mu3      = mu_temp;
      Eigen::MatrixXf mu3_plus = (N3 * mu3 + x) / (N3 + 1.0);
      Eigen::MatrixXf Sigma3   = Sigma_temp;
      // # (3*Sigma1 + 3*(m1-m2).transpose()*(m1-m2) + (x4-m2).transpose()*(x4-m2)) / 4
      mu_temp    = mu3_plus;
      Sigma_temp = Sigma3 + N3 * (mu3 - mu3_plus).transpose() * (mu3 - mu3_plus) + (x - mu3_plus).transpose() * (x - mu3_plus);
   }

   float glda_posterior_predictive(const int& place,
                                   const int& topic,
                                   const float& nu, const float& kappa,
                                   std::unordered_map <int, float>& topic2num_temp,
                                   std::unordered_map <int, Eigen::MatrixXf>& topic2mu_temp,
                                   std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_temp,
                                   const int& log_true, const int& verbose) {
      //std::vector<int> place_vec = doc2place_vec[doc];
      std::vector <int> place_level_vec;
      place_level_vec.push_back(place);

      // KOKOGA MONDAI
      float n = 0;
      float m = 0;
      n = float(topic2num_temp[topic]);   // - float(place_vec.size());
      m = float(place_level_vec.size());

      // Actually we have to re calculate mu and sigma
      float log_prob           = 0.0;
      float prob               = 0.0;
      float first              = 0.0;
      float second_nume        = 0.0;
      float second_denom       = 0.0;
      float third_nume_inside  = 0.0;
      float third_denom_inside = 0.0;
      float third_nume         = 0.0;
      float third_denom        = 0.0;
      float fourth_nume        = 0.0;
      float fourth_denom       = 0.0;
      // Choleskey fast
      std::string method = "llt";

      // KOKO
      first = 0 - std::log(float(place_level_vec.size()) * embedding_dimension / 2.0 * std::log(M_PI));
      if (verbose == 1) {
         std::cout << "  first " << first << std::endl;
      }

      for (int d = 0; d < embedding_dimension; ++d) {
         second_nume  = second_nume + std::lgamma((nu + n + m - float(d)) / 2.0);
         second_denom = second_denom + std::lgamma((nu + n - float(d)) / 2.0);
      }

      if (verbose == 1) {
         std::cout << "  second_nume " << second_nume << std::endl;
         std::cout << "  second_denom " << second_denom << std::endl;
      }

      std::unordered_map <int, float> thres2third_nume_inside;

      if (1 == 1) {
         //if(topic2third_nume_inside.find(topic)==topic2third_nume_inside.end()){
         Eigen::MatrixXf mu           = topic2mu_temp[topic];
         Eigen::MatrixXf Sigma        = topic2Sigma_temp[topic];
         Eigen::MatrixXf Sigma_stable = Psi + Sigma + ((kappa * n) / (kappa + n)) * (mu - embedding_center).transpose() * (mu - embedding_center);
         Sigma_stable = rescale * Sigma_stable;
         if (method == "llt") {
            Eigen::LLT <Eigen::MatrixXf> llt;
            llt.compute(Sigma_stable);
            auto& U = llt.matrixL();
            for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
               if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
                  third_nume_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
               }else{
                  //third_nume_inside += 2*(std::log(U(i, i)/rescale));
                  third_nume_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
               }
               thres2third_nume_inside[i + 1] = third_nume_inside;
            }
         }        //if(method == "llt"){
         if (verbose == 1) {
            std::cout << "third_nume_inside " << third_nume_inside << std::endl;
         }
      }else{
      }

      // third_denom
      float           N_temp     = topic2num_temp[topic];
      Eigen::MatrixXf mu_temp    = topic2mu_temp[topic];
      Eigen::MatrixXf Sigma_temp = topic2Sigma_temp[topic];

      int hantei = 0;
      if (N_temp == 0) {
         hantei = 1;
      }

      for (int i = 0; i < int(place_level_vec.size()); ++i) {
         int             alter_word = doc_word_topic(place_level_vec[i], 1);
         Eigen::MatrixXf alter_x    = embedding_matrix.row(alter_word);
         if (hantei == 0) {
            ghlda_update_mu_sigma(topic, alter_x, N_temp, mu_temp, Sigma_temp);
         }else{
            N_temp += 1;
            mu_temp = mu_temp + alter_x;
         }
      }

      if (hantei == 1 && N_temp > 0) {
         mu_temp = mu_temp / N_temp;
      }

      float value  = (kappa * n + m);
      float value2 = (kappa + n + m);
      float value3 = ((kappa * n + m) / (kappa + n + m));

      Eigen::MatrixXf Sigma_stable = Psi + Sigma_temp + ((kappa * n + m) / (kappa + n + m)) * (mu_temp - embedding_center).transpose() * (mu_temp - embedding_center);
      Sigma_stable = rescale * Sigma_stable;
      if (method == "llt") {
         Eigen::LLT <Eigen::MatrixXf> llt;
         llt.compute(Sigma_stable);
         auto& U = llt.matrixL();
         for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
            if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
               third_denom_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
            }else{
               third_denom_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
            }
         }
         third_nume  = ((nu + n) / 2.0) * third_nume_inside;
         third_denom = ((nu + m + n) / 2.0) * third_denom_inside;
      }

      if (verbose == 1) {
         std::cout << "  value " << value << std::endl;
         std::cout << "  value2 " << value2 << std::endl;
         std::cout << "  value3 " << value3 << std::endl;
         std::cout << "  third_nume " << third_nume << std::endl;
         std::cout << "  third_denom " << third_denom << std::endl;
      }

      fourth_nume  = (embedding_dimension / 2.0) * std::log(kappa + n);
      fourth_denom = (embedding_dimension / 2.0) * std::log(kappa + n + m);

      if (verbose == 1) {
         std::cout << "  n " << n << std::endl;
         std::cout << "  m " << m << std::endl;
         std::cout << "  fourth_nume " << fourth_nume << std::endl;
         std::cout << "  fourth_denom " << fourth_denom << std::endl;
      }

      log_prob = first + second_nume - second_denom + third_nume - third_denom + fourth_nume - fourth_denom;
      log_prob = log_prob - inner_rescale;
      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      return(prob);
   }

   void glda_collapsed_gibbs_sample_parallel(const int num_iteration, const int parallel_loop, int num_threads) {
      stop_hantei = 0;

      if (num_threads < 1) {
         num_threads = omp_get_max_threads();
         std::cout << "Using " << num_threads << " threads" << std::endl;
      }

      omp_set_num_threads(num_threads);

      int max_num_threads = omp_get_max_threads();
      const gsl_rng_type *T;
      gsl_rng **          r;
      gsl_rng_env_setup();
      T = gsl_rng_mt19937;
      r = (gsl_rng **)malloc(max_num_threads * sizeof(gsl_rng *));
      std::random_device rand_device;
      std::mt19937       mt(rand_device());
      for (int i = 0; i < max_num_threads; i++) {
         r[i] = gsl_rng_alloc(T);
         gsl_rng_set(r[i], mt());
      }

      for (int itr = 0; itr < num_iteration; ++itr) {
         if (stop_hantei == 1) {
            std::cout << "ERROR: COULD NOT SAMPLE TOPIC" << std::endl;
            break;
         }

         Eigen::MatrixXi doc_word_topic_update = Eigen::MatrixXi::Zero(parallel_loop, 2);

         #pragma omp parallel for
         for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
            int thread_id = omp_get_thread_num();
            int place     = int(floor(float(doc_word_topic.rows()) * gsl_rng_uniform(r[thread_id])));
            int doc       = doc_word_topic(place, 0);
            int word      = doc_word_topic(place, 1);
            //int old_topic = doc_word_topic(place, 2);

            // word embedding : x (1,embedding_dimension)
            Eigen::MatrixXf x = embedding_matrix.row(word);

            // Create super topic ratio
            std::vector <float> topic_ratio;
            std::vector <float> ratio;

            //create sub topic ratio
            for (int i = 0; i < num_topics; ++i) {
               float temp = alpha_topic_vec(i) + doc2topic_vec[doc](i, 0);
               topic_ratio.push_back(temp);
            }

            // create ratio
            float max_val = -9999999999999;
            std::unordered_map <int, float> multi_t_prob;
            for (int i = 0; i < num_topics; i++) {
               multi_t_prob[i] = glda_posterior_predictive(place, i, nu, kappa, topic2num, topic2mu, topic2Sigma, 1, 0);
            }

            for (int i = 0; i < num_topics; ++i) {
               float temp = std::log(topic_ratio[i]) + multi_t_prob[i];
               ratio.push_back(temp);
               if (max_val < temp) {
                  max_val = temp;
               }
            }

            float agg_topic = 0;
            for (int i = 0; i < int(ratio.size()); ++i) {
               ratio[i]   = ratio[i] - max_val;
               ratio[i]   = std::exp(ratio[i]);
               agg_topic += ratio[i];
            }

            // sample
            double prob[ratio.size()]; // Probability array
            for (int i = 0; i < int(ratio.size()); ++i) {
               prob[i] = double(ratio[i] / agg_topic);
            }
            unsigned int mult_op[ratio.size()];
            int          num_sample = 1;
            gsl_ran_multinomial(r[thread_id], ratio.size(), num_sample, prob, mult_op);
            int new_topic = -1;
            for (int i = 0; i < int(ratio.size()); ++i) {
               if (mult_op[i] == 1) {
                  new_topic = i;
                  break;
               }
            }

            #pragma omp critical
            {
               doc_word_topic_update(itr_inner, 0) = place;
               doc_word_topic_update(itr_inner, 1) = new_topic;
            } //#pragma omp critical
         }    //for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {


         // Update assignments tables parameters
         for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
            int place     = doc_word_topic_update(itr_inner, 0);
            int new_topic = doc_word_topic_update(itr_inner, 1);
            int doc       = doc_word_topic(place, 0);
            int word      = doc_word_topic(place, 1);
            int topic     = doc_word_topic(place, 2);

            if (new_topic < 0) {
               std::cout << "ERROR" << std::endl;
               stop_hantei = 1;
               break;
            }

            // word embedding : x (1,embedding_dimension)
            Eigen::MatrixXf x = embedding_matrix.row(word);

            if (doc2topic_vec[doc](topic, 0) > 0.0) {
               // Subtract tables
               doc2topic_vec[doc](topic, 0) = doc2topic_vec[doc](topic, 0) - 1.0;

               // Update tables
               doc2topic_vec[doc](new_topic, 0) = doc2topic_vec[doc](new_topic, 0) + 1.0;

               // Subtract parameters
               glda_update_parameters_subtract(topic, word);

               // Update parameters
               glda_update_parameters_add(new_topic, word);

               // Update assignments
               doc_word_topic(place, 2) = new_topic;
            }
         } //for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner)
      }    //for (int itr = 0; itr < num_iteration; ++itr)

      // Free memory
      for (int i = 0; i < max_num_threads; i++) {
         gsl_rng_free(r[i]);
      }
   }

   float glda_posterior_predictive_second_nume_denom(const int& topic,
                                                     const float& nu, const float& kappa,
                                                     std::unordered_map <int, float>& topic2num_temp,
                                                     std::unordered_map <int, Eigen::MatrixXf>& topic2mu_temp,
                                                     std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_temp,
                                                     const int& verbose) {
      float n            = float(topic2num_temp[topic]);
      float m            = 1.0;
      float second_nume  = 0.0;
      float second_denom = 0.0;

      for (int d = 0; d < embedding_dimension; ++d) {
         second_nume  = second_nume + std::lgamma((nu + n + m - float(d)) / 2.0);
         second_denom = second_denom + std::lgamma((nu + n - float(d)) / 2.0);
      }
      float second_nume_denom = second_nume - second_denom;
      return(second_nume_denom);
   }

   float glda_posterior_predictive_third_nume(const int& topic,
                                              const float& nu, const float& kappa,
                                              std::unordered_map <int, float>& topic2num_temp,
                                              std::unordered_map <int, Eigen::MatrixXf>& topic2mu_temp,
                                              std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_temp,
                                              const int& verbose) {
      // KOKOGA MONDAI
      float n = 0;

      n = float(topic2num_temp[topic]);   // - float(place_vec.size());

      float third_nume_inside = 0.0;
      float third_nume        = 0.0;
      // Choleskey fast
      std::string method = "llt";

      //if(topic2third_nume_inside.find(topic)==topic2third_nume_inside.end()){
      Eigen::MatrixXf mu           = topic2mu_temp[topic];
      Eigen::MatrixXf Sigma        = topic2Sigma_temp[topic];
      Eigen::MatrixXf Sigma_stable = Psi + Sigma + ((kappa * n) / (kappa + n)) * (mu - embedding_center).transpose() * (mu - embedding_center);
      Sigma_stable = rescale * Sigma_stable;
      if (method == "llt") {
         Eigen::LLT <Eigen::MatrixXf> llt;
         llt.compute(Sigma_stable);
         auto& U = llt.matrixL();
         for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
            if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
               third_nume_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
            }else{
               //third_nume_inside += 2*(std::log(U(i, i)/rescale));
               third_nume_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
            }
         }        //if(method == "llt"){
         if (verbose == 1) {
            std::cout << "third_nume_inside " << third_nume_inside << std::endl;
         }
      }else{
      }

      third_nume = ((nu + n) / 2.0) * third_nume_inside;
      return(third_nume);
   }

   float glda_posterior_predictive_fast(const int& place,
                                        const int& topic,
                                        const float& nu, const float& kappa,
                                        std::unordered_map <int, float>& topic2num_temp,
                                        std::unordered_map <int, Eigen::MatrixXf>& topic2mu_temp,
                                        std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_temp,
                                        std::unordered_map <int, float>& topic2second_nume_denom,
                                        std::unordered_map <int, float>& topic2third_nume,
                                        const int& log_true, const int& verbose) {
      //std::vector<int> place_vec = doc2place_vec[doc];
      std::vector <int> place_level_vec;
      place_level_vec.push_back(place);

      // KOKOGA MONDAI
      float n = 0;
      float m = 0;
      n = float(topic2num_temp[topic]);   // - float(place_vec.size());
      m = float(place_level_vec.size());

      // Actually we have to re calculate mu and sigma
      float log_prob           = 0.0;
      float prob               = 0.0;
      float first              = 0.0;
      float second_nume_denom  = 0.0;
      float third_denom_inside = 0.0;
      float third_nume         = 0.0;
      float third_denom        = 0.0;
      float fourth_nume        = 0.0;
      float fourth_denom       = 0.0;
      // Choleskey fast
      std::string method = "llt";

      // KOKO
      first = 0 - std::log(float(place_level_vec.size()) * embedding_dimension / 2.0 * std::log(M_PI));
      if (verbose == 1) {
         std::cout << "  first " << first << std::endl;
      }

      second_nume_denom = topic2second_nume_denom[topic];

      // third_denom
      float           N_temp     = topic2num_temp[topic];
      Eigen::MatrixXf mu_temp    = topic2mu_temp[topic];
      Eigen::MatrixXf Sigma_temp = topic2Sigma_temp[topic];

      int hantei = 0;
      if (N_temp == 0) {
         hantei = 1;
      }

      for (int i = 0; i < int(place_level_vec.size()); ++i) {
         int             alter_word = doc_word_topic(place_level_vec[i], 1);
         Eigen::MatrixXf alter_x    = embedding_matrix.row(alter_word);
         if (hantei == 0) {
            ghlda_update_mu_sigma(topic, alter_x, N_temp, mu_temp, Sigma_temp);
         }else{
            N_temp += 1;
            mu_temp = mu_temp + alter_x;
         }
      }

      if (hantei == 1 && N_temp > 0) {
         mu_temp = mu_temp / N_temp;
      }

      float value  = (kappa * n + m);
      float value2 = (kappa + n + m);
      float value3 = ((kappa * n + m) / (kappa + n + m));

      Eigen::MatrixXf Sigma_stable = Psi + Sigma_temp + ((kappa * n + m) / (kappa + n + m)) * (mu_temp - embedding_center).transpose() * (mu_temp - embedding_center);
      Sigma_stable = rescale * Sigma_stable;
      if (method == "llt") {
         Eigen::LLT <Eigen::MatrixXf> llt;
         llt.compute(Sigma_stable);
         auto& U = llt.matrixL();
         for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
            if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
               third_denom_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
            }else{
               third_denom_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
            }
         }
         third_nume  = topic2third_nume[topic];
         third_denom = ((nu + m + n) / 2.0) * third_denom_inside;
      }

      if (verbose == 1) {
         std::cout << "  value " << value << std::endl;
         std::cout << "  value2 " << value2 << std::endl;
         std::cout << "  value3 " << value3 << std::endl;
         std::cout << "  third_nume " << third_nume << std::endl;
         std::cout << "  third_denom " << third_denom << std::endl;
      }

      fourth_nume  = (embedding_dimension / 2.0) * std::log(kappa + n);
      fourth_denom = (embedding_dimension / 2.0) * std::log(kappa + n + m);

      if (verbose == 1) {
         std::cout << "  n " << n << std::endl;
         std::cout << "  m " << m << std::endl;
         std::cout << "  fourth_nume " << fourth_nume << std::endl;
         std::cout << "  fourth_denom " << fourth_denom << std::endl;
      }

      log_prob = first + second_nume_denom + third_nume - third_denom + fourth_nume - fourth_denom;
      log_prob = log_prob - inner_rescale;
      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      return(prob);
   }

   void glda_collapsed_gibbs_sample_parallel_fast(const int num_iteration, const int parallel_loop, int num_threads) {
      stop_hantei = 0;

      if (num_threads < 1) {
         num_threads = omp_get_max_threads();
         std::cout << "Using " << num_threads << " threads" << std::endl;
      }

      omp_set_num_threads(num_threads);

      int max_num_threads = omp_get_max_threads();
      const gsl_rng_type *T;
      gsl_rng **          r;
      gsl_rng_env_setup();
      T = gsl_rng_mt19937;
      r = (gsl_rng **)malloc(max_num_threads * sizeof(gsl_rng *));
      std::random_device rand_device;
      std::mt19937       mt(rand_device());
      for (int i = 0; i < max_num_threads; i++) {
         r[i] = gsl_rng_alloc(T);
         gsl_rng_set(r[i], mt());
      }

      for (int itr = 0; itr < num_iteration; ++itr) {
         if (stop_hantei == 1) {
            std::cout << "ERROR: COULD NOT SAMPLE TOPIC" << std::endl;
            break;
         }

         std::unordered_map <int, float> topic2second_nume_denom, topic2third_nume;
         #pragma omp parallel for
         for (int i = 0; i < num_topics; ++i) {
            float second_nume_denom_t = glda_posterior_predictive_second_nume_denom(i, nu, kappa, topic2num, topic2mu, topic2Sigma, 0);
            float third_nume_t        = glda_posterior_predictive_third_nume(i, nu, kappa, topic2num, topic2mu, topic2Sigma, 0);
            #pragma omp critical
            {
               topic2second_nume_denom[i] = second_nume_denom_t;
               topic2third_nume[i]        = third_nume_t;
            }
         }

         Eigen::MatrixXi doc_word_topic_update = Eigen::MatrixXi::Zero(parallel_loop, 2);

         #pragma omp parallel for
         for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
            int thread_id = omp_get_thread_num();
            int place     = int(floor(float(doc_word_topic.rows()) * gsl_rng_uniform(r[thread_id])));
            int doc       = doc_word_topic(place, 0);
            int word      = doc_word_topic(place, 1);
            //int old_topic = doc_word_topic(place, 2);

            // word embedding : x (1,embedding_dimension)
            Eigen::MatrixXf x = embedding_matrix.row(word);

            // Create super topic ratio
            std::vector <float> topic_ratio;
            std::vector <float> ratio;

            //create sub topic ratio
            for (int i = 0; i < num_topics; ++i) {
               float temp = alpha_topic_vec(i) + doc2topic_vec[doc](i, 0);
               topic_ratio.push_back(temp);
            }

            // create ratio
            float max_val = -9999999999999;
            std::unordered_map <int, float> multi_t_prob;
            for (int i = 0; i < num_topics; i++) {
               multi_t_prob[i] = glda_posterior_predictive_fast(place, i, nu, kappa, topic2num, topic2mu, topic2Sigma, topic2second_nume_denom, topic2third_nume, 1, 0);
            }

            for (int i = 0; i < num_topics; ++i) {
               float temp = std::log(topic_ratio[i]) + multi_t_prob[i];
               ratio.push_back(temp);
               if (max_val < temp) {
                  max_val = temp;
               }
            }

            float agg_topic = 0;
            for (int i = 0; i < int(ratio.size()); ++i) {
               ratio[i]   = ratio[i] - max_val;
               ratio[i]   = std::exp(ratio[i]);
               agg_topic += ratio[i];
            }

            // sample
            double prob[ratio.size()]; // Probability array
            for (int i = 0; i < int(ratio.size()); ++i) {
               prob[i] = double(ratio[i] / agg_topic);
            }
            unsigned int mult_op[ratio.size()];
            int          num_sample = 1;
            gsl_ran_multinomial(r[thread_id], ratio.size(), num_sample, prob, mult_op);
            int new_topic = -1;
            for (int i = 0; i < int(ratio.size()); ++i) {
               if (mult_op[i] == 1) {
                  new_topic = i;
                  break;
               }
            }

            #pragma omp critical
            {
               doc_word_topic_update(itr_inner, 0) = place;
               doc_word_topic_update(itr_inner, 1) = new_topic;
            } //#pragma omp critical
         }    //for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {


         // Update assignments tables parameters
         for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
            int place     = doc_word_topic_update(itr_inner, 0);
            int new_topic = doc_word_topic_update(itr_inner, 1);
            int doc       = doc_word_topic(place, 0);
            int word      = doc_word_topic(place, 1);
            int topic     = doc_word_topic(place, 2);

            if (new_topic < 0) {
               std::cout << "ERROR" << std::endl;
               stop_hantei = 1;
               break;
            }

            // word embedding : x (1,embedding_dimension)
            Eigen::MatrixXf x = embedding_matrix.row(word);

            if (doc2topic_vec[doc](topic, 0) > 0.0) {
               // Subtract tables
               doc2topic_vec[doc](topic, 0) = doc2topic_vec[doc](topic, 0) - 1.0;

               // Update tables
               doc2topic_vec[doc](new_topic, 0) = doc2topic_vec[doc](new_topic, 0) + 1.0;

               // Subtract parameters
               glda_update_parameters_subtract(topic, word);

               // Update parameters
               glda_update_parameters_add(new_topic, word);

               // Update assignments
               doc_word_topic(place, 2) = new_topic;
            }
         } //for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner)
      }    //for (int itr = 0; itr < num_iteration; ++itr)

      // Free memory
      for (int i = 0; i < max_num_threads; i++) {
         gsl_rng_free(r[i]);
      }
   }

   float glda_posterior_predictive_second_nume_denom_faster(const int& topic,
                                                            const float& nu, const float& kappa,
                                                            std::unordered_map <int, float>& topic2num_temp,
                                                            std::unordered_map <int, Eigen::MatrixXf>& topic2mu_temp,
                                                            std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_temp,
                                                            const int& verbose) {
      float n            = float(topic2num_temp[topic]);
      float m            = 1.0;
      float second_nume  = 0.0;
      float second_denom = 0.0;

      for (int d = 0; d < embedding_dimension; ++d) {
         second_nume  = second_nume + fastlgamma((nu + n + m - float(d)) / 2.0);
         second_denom = second_denom + fastlgamma((nu + n - float(d)) / 2.0);
      }
      float second_nume_denom = second_nume - second_denom;
      return(second_nume_denom);
   }

   void glda_collapsed_gibbs_sample_parallel_faster(const int num_iteration, const int parallel_loop, int num_threads) {
      stop_hantei = 0;

      if (num_threads < 1) {
         num_threads = omp_get_max_threads();
         std::cout << "Using " << num_threads << " threads" << std::endl;
      }

      omp_set_num_threads(num_threads);

      int max_num_threads = omp_get_max_threads();
      const gsl_rng_type *T;
      gsl_rng **          r;
      gsl_rng_env_setup();
      T = gsl_rng_mt19937;
      r = (gsl_rng **)malloc(max_num_threads * sizeof(gsl_rng *));
      std::random_device rand_device;
      std::mt19937       mt(rand_device());
      for (int i = 0; i < max_num_threads; i++) {
         r[i] = gsl_rng_alloc(T);
         gsl_rng_set(r[i], mt());
      }

      for (int itr = 0; itr < num_iteration; ++itr) {
         if (stop_hantei == 1) {
            std::cout << "ERROR: COULD NOT SAMPLE TOPIC" << std::endl;
            break;
         }

         std::unordered_map <int, float> topic2second_nume_denom, topic2third_nume;
         #pragma omp parallel for
         for (int i = 0; i < num_topics; ++i) {
            float second_nume_denom_t = glda_posterior_predictive_second_nume_denom_faster(i, nu, kappa, topic2num, topic2mu, topic2Sigma, 0);
            float third_nume_t        = glda_posterior_predictive_third_nume(i, nu, kappa, topic2num, topic2mu, topic2Sigma, 0);
            #pragma omp critical
            {
               topic2second_nume_denom[i] = second_nume_denom_t;
               topic2third_nume[i]        = third_nume_t;
            }
         }

         Eigen::MatrixXi doc_word_topic_update = Eigen::MatrixXi::Zero(parallel_loop, 2);

         #pragma omp parallel for
         for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
            int thread_id = omp_get_thread_num();
            int place     = int(floor(float(doc_word_topic.rows()) * gsl_rng_uniform(r[thread_id])));
            int doc       = doc_word_topic(place, 0);
            int word      = doc_word_topic(place, 1);
            //int old_topic = doc_word_topic(place, 2);

            // word embedding : x (1,embedding_dimension)
            Eigen::MatrixXf x = embedding_matrix.row(word);

            std::vector <float> topic_ratio;
            std::vector <float> ratio;

            for (int i = 0; i < num_topics; ++i) {
               float temp = alpha_topic_vec(i) + doc2topic_vec[doc](i, 0);
               topic_ratio.push_back(temp);
            }

            // create ratio
            float max_val = -9999999999999;
            std::unordered_map <int, float> multi_t_prob;
            for (int i = 0; i < num_topics; i++) {
               multi_t_prob[i] = glda_posterior_predictive_fast(place, i, nu, kappa, topic2num, topic2mu, topic2Sigma, topic2second_nume_denom, topic2third_nume, 1, 0);
            }

            for (int i = 0; i < num_topics; ++i) {
               float temp = std::log(topic_ratio[i]) + multi_t_prob[i];
               ratio.push_back(temp);
               if (max_val < temp) {
                  max_val = temp;
               }
            }

            float agg_topic = 0;
            for (int i = 0; i < int(ratio.size()); ++i) {
               ratio[i]   = ratio[i] - max_val;
               ratio[i]   = std::exp(ratio[i]);
               agg_topic += ratio[i];
            }

            // sample
            double prob[ratio.size()]; // Probability array
            for (int i = 0; i < int(ratio.size()); ++i) {
               prob[i] = double(ratio[i] / agg_topic);
            }
            unsigned int mult_op[ratio.size()];
            int          num_sample = 1;
            gsl_ran_multinomial(r[thread_id], ratio.size(), num_sample, prob, mult_op);
            int new_topic = -1;
            for (int i = 0; i < int(ratio.size()); ++i) {
               if (mult_op[i] == 1) {
                  new_topic = i;
                  break;
               }
            }

            #pragma omp critical
            {
               doc_word_topic_update(itr_inner, 0) = place;
               doc_word_topic_update(itr_inner, 1) = new_topic;
            } //#pragma omp critical
         }    //for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {


         // Update assignments tables parameters
         for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
            int place     = doc_word_topic_update(itr_inner, 0);
            int new_topic = doc_word_topic_update(itr_inner, 1);
            int doc       = doc_word_topic(place, 0);
            int word      = doc_word_topic(place, 1);
            int topic     = doc_word_topic(place, 2);

            if (new_topic < 0) {
               std::cout << "ERROR" << std::endl;
               stop_hantei = 1;
               break;
            }

            // word embedding : x (1,embedding_dimension)
            Eigen::MatrixXf x = embedding_matrix.row(word);

            if (doc2topic_vec[doc](topic, 0) > 0.0) {
               // Subtract tables
               doc2topic_vec[doc](topic, 0) = doc2topic_vec[doc](topic, 0) - 1.0;

               // Update tables
               doc2topic_vec[doc](new_topic, 0) = doc2topic_vec[doc](new_topic, 0) + 1.0;

               // Subtract parameters
               glda_update_parameters_subtract(topic, word);

               // Update parameters
               glda_update_parameters_add(new_topic, word);

               // Update assignments
               doc_word_topic(place, 2) = new_topic;
            }
         } //for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner)
      }    //for (int itr = 0; itr < num_iteration; ++itr)

      // Free memory
      for (int i = 0; i < max_num_threads; i++) {
         gsl_rng_free(r[i]);
      }
   }

   std::unordered_map <int, std::string> id2word;
   std::unordered_map <int, Eigen::MatrixXf> topic2word_prob;
   std::unordered_map <int, Eigen::MatrixXf> topic2word_prob_debug;
   std::unordered_map <int, Eigen::MatrixXf> doc_test2heldout_prob;

   void create_doc2place_vec_test() {
      num_docs_test = 0;
      doc2place_vec_test.clear();
      for (int i = 0; i < doc_word_topic_test.rows(); ++i) {
         if (doc2place_vec_test.find(doc_word_topic_test(i, 0)) == doc2place_vec_test.end()) {
            std::vector <int> temp_vec;
            temp_vec.push_back(i);
            doc2place_vec_test[doc_word_topic_test(i, 0)] = temp_vec;
            num_docs_test += 1;
         }else{
            doc2place_vec_test[doc_word_topic_test(i, 0)].push_back(i);
         }
      }
   }

   float glda_posterior_predictive_test(const Eigen::MatrixXi doc_word_topic_thread_temp, const int& place,
                                        const int& topic,
                                        const float& nu, const float& kappa,
                                        std::unordered_map <int, float>& topic2num_temp,
                                        std::unordered_map <int, Eigen::MatrixXf>& topic2mu_temp,
                                        std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_temp,
                                        const int& log_true, const int& verbose) {
      std::vector <int> place_level_vec;
      place_level_vec.push_back(place);
      float n = 0;
      float m = 0;
      n = float(topic2num_temp[topic]);
      m = float(place_level_vec.size());

      // Actually we have to re calculate mu and sigma
      float log_prob           = 0.0;
      float prob               = 0.0;
      float first              = 0.0;
      float second_nume        = 0.0;
      float second_denom       = 0.0;
      float third_nume_inside  = 0.0;
      float third_denom_inside = 0.0;
      float third_nume         = 0.0;
      float third_denom        = 0.0;
      float fourth_nume        = 0.0;
      float fourth_denom       = 0.0;
      // Choleskey fast
      std::string method = "llt";

      // KOKO
      first = 0 - std::log(float(place_level_vec.size()) * embedding_dimension / 2.0 * std::log(M_PI));
      if (verbose == 1) {
         std::cout << "  first " << first << std::endl;
      }

      for (int d = 0; d < embedding_dimension; ++d) {
         second_nume  = second_nume + std::lgamma((nu + n + m - float(d)) / 2.0);
         second_denom = second_denom + std::lgamma((nu + n - float(d)) / 2.0);
      }

      if (verbose == 1) {
         std::cout << "  second_nume " << second_nume << std::endl;
         std::cout << "  second_denom " << second_denom << std::endl;
      }

      std::unordered_map <int, float> thres2third_nume_inside;

      if (1 == 1) {
         Eigen::MatrixXf mu           = topic2mu_temp[topic];
         Eigen::MatrixXf Sigma        = topic2Sigma_temp[topic];
         Eigen::MatrixXf Sigma_stable = Psi + Sigma + ((kappa * n) / (kappa + n)) * (mu - embedding_center).transpose() * (mu - embedding_center);
         Sigma_stable = rescale * Sigma_stable;
         if (method == "llt") {
            Eigen::LLT <Eigen::MatrixXf> llt;
            llt.compute(Sigma_stable);
            auto& U = llt.matrixL();
            for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
               if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
                  third_nume_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
               }else{
                  third_nume_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
               }
               thres2third_nume_inside[i + 1] = third_nume_inside;
            }
         }        //if(method == "llt"){
         if (verbose == 1) {
            std::cout << "third_nume_inside " << third_nume_inside << std::endl;
         }
      }

      // third_denom
      float           N_temp     = topic2num_temp[topic];
      Eigen::MatrixXf mu_temp    = topic2mu_temp[topic];
      Eigen::MatrixXf Sigma_temp = topic2Sigma_temp[topic];

      int hantei = 0;
      if (N_temp == 0) {
         hantei = 1;
      }

      for (int i = 0; i < int(place_level_vec.size()); ++i) {
         int             alter_word = doc_word_topic_thread_temp(place_level_vec[i], 1);
         Eigen::MatrixXf alter_x    = embedding_matrix.row(alter_word);
         if (hantei == 0) {
            ghlda_update_mu_sigma(topic, alter_x, N_temp, mu_temp, Sigma_temp);
         }else{
            N_temp += 1;
            mu_temp = mu_temp + alter_x;
         }
      }

      if (hantei == 1 && N_temp > 0) {
         mu_temp = mu_temp / N_temp;
      }

      float value  = (kappa * n + m);
      float value2 = (kappa + n + m);
      float value3 = ((kappa * n + m) / (kappa + n + m));

      Eigen::MatrixXf Sigma_stable = Psi + Sigma_temp + ((kappa * n + m) / (kappa + n + m)) * (mu_temp - embedding_center).transpose() * (mu_temp - embedding_center);
      Sigma_stable = rescale * Sigma_stable;
      if (method == "llt") {
         Eigen::LLT <Eigen::MatrixXf> llt;
         llt.compute(Sigma_stable);
         auto& U = llt.matrixL();
         for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
            if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
               third_denom_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
            }else{
               third_denom_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
            }
         }
         third_nume  = ((nu + n) / 2.0) * third_nume_inside;
         third_denom = ((nu + m + n) / 2.0) * third_denom_inside;
      }

      if (verbose == 1) {
         std::cout << "  value " << value << std::endl;
         std::cout << "  value2 " << value2 << std::endl;
         std::cout << "  value3 " << value3 << std::endl;
         std::cout << "  third_nume " << third_nume << std::endl;
         std::cout << "  third_denom " << third_denom << std::endl;
      }

      fourth_nume  = (embedding_dimension / 2.0) * std::log(kappa + n);
      fourth_denom = (embedding_dimension / 2.0) * std::log(kappa + n + m);

      if (verbose == 1) {
         std::cout << "  n " << n << std::endl;
         std::cout << "  m " << m << std::endl;
         std::cout << "  fourth_nume " << fourth_nume << std::endl;
         std::cout << "  fourth_denom " << fourth_denom << std::endl;
      }

      log_prob = first + second_nume - second_denom + third_nume - third_denom + fourth_nume - fourth_denom;
      log_prob = log_prob - inner_rescale;
      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      return(prob);
   }

   void create_topic2word_map() {

      for (auto itr = topic2num.begin(); itr != topic2num.end(); ++itr) {
          topic2num[itr->first]      = 0.0;
          topic2word_map[itr->first] = zero_topic2word_map;
      }

      topic2num[-1]      = 0.0;
      topic2word_map[-1] = zero_topic2word_map;

      for (int i = 0; i < doc_word_topic.rows(); ++i) {
          //int doc   = doc_word_topic(i, 0);
          int word  = doc_word_topic(i, 1);
          //int level = doc_word_topic(i, 3);
          int topic = doc_word_topic(i, 4);

          if (topic2word_map.find(topic) == topic2word_map.end()) {
              topic2word_map[topic] = zero_topic2word_map;
          }
          if (topic2word_map[topic].find(word) == topic2word_map[topic].end()) {
              topic2word_map[topic][word] = 1.0;
          }else{
              topic2word_map[topic][word] += 1.0;
          }

          if (topic != -1) {
              // Create Parameters
              if (topic2num.find(topic) == topic2num.end()) {
                topic2num[topic] = 1.0;
              }else{
                topic2num[topic] += 1.0;
              }
         }
      }// for (int i = 0; i < doc_word_topic.rows(); ++i) {
   }

   void create_topic2word_prob(
      std::unordered_map <int, Eigen::MatrixXf>& topic2word_prob,
      std::unordered_map <int, Eigen::MatrixXf>& topic2mu_thread,
      std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_thread,
      float eta_thres, std::unordered_map <int, int>& update_topic_thread) {
      int size = 0;

      for (auto itr = id2word.begin(); itr != id2word.end(); ++itr) {
         size += 1;
      }
      for (auto itr = topic2mu.begin(); itr != topic2mu.end(); ++itr) {
         int topic = itr->first;
         if (update_topic_thread.find(topic) != update_topic_thread.end()) {
            Eigen::MatrixXf word_prob = Eigen::MatrixXf::Zero(size, 1);
            float           max_val   = -999999999999;
            for (auto itr2 = id2word.begin(); itr2 != id2word.end(); ++itr2) {
               int             word = itr2->first;
               Eigen::MatrixXf x    = embedding_matrix.row(word);
               float           prob = prob_mass_mvt(x, nu, topic2mu_thread[topic], topic2Sigma_thread[topic], 1, 0);
               if (max_val < prob) {
                  max_val = prob;
               }
               word_prob(word, 0) = prob;
            }
            float agg = 0.0;
            for (auto itr2 = id2word.begin(); itr2 != id2word.end(); ++itr2) {
               int word = itr2->first;
               word_prob(word, 0) = std::exp(word_prob(word, 0) - max_val) + eta_thres;
               agg += word_prob(word, 0);
            }
            for (auto itr2 = id2word.begin(); itr2 != id2word.end(); ++itr2) {
               int word = itr2->first;
               word_prob(word, 0) = word_prob(word, 0) / agg;
            }
            topic2word_prob[topic] = word_prob;
         }
      }
   }

   void glda_evaluate_held_out_log_likelihood(const std::vector <int> doc_test_ids, const int num_iteration, const int burn_in, const float eta_thres, const int approx, int num_threads, int verbose) {

      if (verbose == 1) {
         std::cout << "Enter glda_evaluate_held_out_log_likelihood" << std::endl;
      }

      if (num_threads < 1) {
         num_threads = omp_get_max_threads();
         std::cout << "Using " << num_threads << " threads" << std::endl;
      }

      omp_set_num_threads(num_threads);
      int max_num_threads = omp_get_max_threads();
      const gsl_rng_type *T;
      gsl_rng **          r;
      gsl_rng_env_setup();
      T = gsl_rng_mt19937;
      r = (gsl_rng **)malloc(max_num_threads * sizeof(gsl_rng *));
      std::random_device rand_device;
      std::mt19937       mt(rand_device());
      for (int i = 0; i < max_num_threads; i++) {
         r[i] = gsl_rng_alloc(T);
         gsl_rng_set(r[i], mt());
      }

      // create_topic2word_prob: Initially we have to update everything
      std::unordered_map <int, int> update_topic;
      for (auto itr = topic2num.begin(); itr != topic2num.end(); ++itr) {
         update_topic[itr->first] = 1;
      }

      if (verbose == 1) {
         std::cout << "Before create_topic2word_prob" << std::endl;
      }

      // version 1
      //ghlda_ncrp_create_topic2word_prob(topic2word_prob, topic2mu, topic2Sigma, eta_thres, update_topic);
      // version 2
      create_topic2word_map();

      if (verbose == 1) {
         std::cout << "Before pragma omp parallel for" << std::endl;
      }

      float counter = 0.0;
      #pragma omp parallel for
      for (int k = 0; k < int(doc_test_ids.size()); ++k) {
         std::unordered_map <int, Eigen::MatrixXf> topic2word_prob_thread = topic2word_prob;
         std::unordered_map <int, std::unordered_map <int, float> > topic2word_map_thread = topic2word_map;
         std::unordered_map <int, int>             update_topic_thread;
         int             doc_test = doc_test_ids[k];
         int             doc_size = int(doc2place_vec_test[doc_test].size());
         Eigen::MatrixXi doc_word_topic_thread = Eigen::MatrixXi::Zero(doc_size, 8);  // take log sum of this
         for (int i = 0; i < doc_size; ++i) {
            int place_t = doc2place_vec_test[doc_test][i];
            doc_word_topic_thread(i, 0) = doc_word_topic_test(place_t, 0);
            doc_word_topic_thread(i, 1) = doc_word_topic_test(place_t, 1);
            doc_word_topic_thread(i, 2) = doc_word_topic_test(place_t, 2);
         }

         if (verbose == 1) {
            std::cout << "Before for (int up_to = 0" << std::endl;
         }

         // for every word position
         Eigen::MatrixXf heldout_prob = Eigen::MatrixXf::Zero(doc_word_topic_thread.rows(), 1);
         std::unordered_map <int, std::unordered_map <int, float> > place2topic_weight_thread;
         for (int up_to = 0; up_to < int(doc_word_topic_thread.rows()); ++up_to) {
            int thread_id = omp_get_thread_num();
            std::unordered_map <int, float>           topic2num_thread   = topic2num;
            std::unordered_map <int, Eigen::MatrixXf> topic2mu_thread    = topic2mu;
            std::unordered_map <int, Eigen::MatrixXf> topic2Sigma_thread = topic2Sigma;
            Eigen::MatrixXf topic_vec_sub = Eigen::MatrixXf::Zero(num_topics, 1);
            Eigen::MatrixXi doc_word_topic_thread_temp = Eigen::MatrixXi::Zero(up_to + 1, 10);

            for (int i = 0; i < (up_to + 1); ++i) {
               doc_word_topic_thread_temp(i, 0) = doc_word_topic_thread(i, 0);
               doc_word_topic_thread_temp(i, 1) = doc_word_topic_thread(i, 1);
               doc_word_topic_thread_temp(i, 2) = doc_word_topic_thread(i, 2);
               int topic = doc_word_topic_thread_temp(i, 2);
               //int word  = doc_word_topic_thread_temp(i, 1);
               if (i == up_to) {
                  topic_vec_sub(topic, 0) += 1.0;
               }else{
                  for (auto itr3 = place2topic_weight_thread[i].begin(); itr3 != place2topic_weight_thread[i].end(); ++itr3) {
                     topic = itr3->first;
                     float val = itr3->second;
                     topic_vec_sub(topic, 0) += val;
                  }
               } //for (auto itr3 = place2topic_weight_thread[i].begin()
            }    //or (int i = 0; i < (up_to + 1); ++i) {

            float agg   = 0;
            float count = 0;
            update_topic_thread.clear();
            std::unordered_map <int, float> topic_weight_thread;
            for (int rep = 0; rep < num_iteration; ++rep) {
               int place     = int(doc_word_topic_thread_temp.rows()) - 1;
               int word      = doc_word_topic_thread_temp(place, 1);
               int old_topic = doc_word_topic_thread_temp(place, 2);
               std::vector <float> topic_ratio;
               std::vector <float> ratio;
               for (int i = 0; i < num_topics; ++i) {
                  float temp = alpha_topic_vec(i) + topic_vec_sub(i, 0);
                  topic_ratio.push_back(temp);
               }
               // create ratio
               float max_val = -9999999999999;
               std::unordered_map <int, float> multi_t_prob;
               for (int i = 0; i < num_topics; i++) {
                  multi_t_prob[i] = glda_posterior_predictive_test(doc_word_topic_thread_temp, place, i, nu, kappa, topic2num_thread, topic2mu_thread, topic2Sigma_thread, 1, 0);
                  float temp = std::log(topic_ratio[i]) + multi_t_prob[i];
                  ratio.push_back(temp);
                  if (max_val < temp) {
                     max_val = temp;
                  }
               }
               float agg_topic = 0;
               for (int i = 0; i < int(ratio.size()); ++i) {
                  ratio[i]   = ratio[i] - max_val;
                  ratio[i]   = std::exp(ratio[i]);
                  agg_topic += ratio[i];
               }
               double prob[ratio.size()];     // Probability array
               for (int i = 0; i < int(ratio.size()); ++i) {
                  prob[i] = double(ratio[i] / agg_topic);
               }
               unsigned int mult_op[ratio.size()];
               int          num_sample = 1;
               gsl_ran_multinomial(r[thread_id], ratio.size(), num_sample, prob, mult_op);
               int new_topic = -1;
               for (int i = 0; i < int(ratio.size()); ++i) {
                  if (mult_op[i] == 1) {
                     new_topic = i;
                     break;
                  }
               }
               if (update_topic_thread.find(new_topic) == update_topic_thread.end()) {
                  update_topic_thread[new_topic] = 1;
               }
               topic_vec_sub(old_topic, 0)         -= 1.0;
               topic_vec_sub(new_topic, 0)         += 1.0;
               doc_word_topic_thread_temp(place, 2) = new_topic;
               doc_word_topic_thread(place, 2)      = new_topic;
               if (rep > burn_in) {
                  update_topic_thread.clear();
                  for (int j = 0; j < int(ratio.size()); ++j) {
                      new_topic = j;
                      if(topic2word_map_thread[new_topic].find(word)!=topic2word_map_thread[new_topic].end()){
                        agg += float(prob[j]) * (topic2word_map_thread[new_topic][word] + beta) / (voc * beta + topic2num_thread[new_topic]);
                      }else{
                        agg += float(prob[j]) * beta / (voc * beta + topic2num_thread[new_topic]);
                      }
                  }
                  count += 1.0;
                  if (topic_weight_thread.find(new_topic) == topic_weight_thread.end()) {
                     topic_weight_thread[new_topic] = 1.0;
                  }else{
                     topic_weight_thread[new_topic] += 1.0;
                  }
               } //if (rep > burn_in) {
            }    //for(int rep = 0;rep < num_iteration;++rep){

            // record heldout_prob
            float prob_2 = agg / count;
            heldout_prob(up_to, 0) = prob_2;

            // Update place2topic_weight_thread
            float agg2 = 0.0;
            for (auto itr3 = topic_weight_thread.begin(); itr3 != topic_weight_thread.end(); ++itr3) {
               agg2 += 1.0;
            }
            for (auto itr3 = topic_weight_thread.begin(); itr3 != topic_weight_thread.end(); ++itr3) {
               itr3->second = itr3->second / agg2;
            }
            place2topic_weight_thread[up_to] = topic_weight_thread;
         }  //for (int up_to = 0; up_to < int(doc_word_topic_sub.rows()); ++up_to) {
         #pragma omp critical
         {
            doc_test2heldout_prob[doc_test] = heldout_prob;
            counter += 1;
            std::cout << "Finished: " << counter / float(doc_test_ids.size()) << std::endl;
         } //#pragma omp critical
      }    //for (int k = 0; k < int(doc_test_ids.size()); ++k)
   }

   // GLDA: END //

   // GHLDA-Fixed START //
   float sample_counter   = 0.0;
   float path_gamma       = 1.0;
   float num_not_assigned = 0.0;
   int max_depth_allowed  = 3;
   float level_gamma      = 1.0;
   float mix_ratio        = 0.2;
   std::unordered_map <int, Eigen::MatrixXf> level2Psi;
   std::unordered_map <int, float> word2cdf;

   int path_level2topic(const int path, const int level) {
      int topic_id = -1;

      if (path_dict.find(path) != path_dict.end()) {
         std::vector <int> v1 = path_dict[path];
         topic_id = v1[level];
      }
      return(topic_id);
   }

   void create_initial_hierarchy(const int initial_depth, const int initial_branch) {
      num_topics = 0;
      num_path   = 0;
      num_depth  = initial_depth;
      topic_dict.clear();
      path_dict.clear();

      std::vector <std::string> temp_string_vec;
      for (int i = 0; i < initial_depth; ++i) {
         if (i == 0) {
            std::string temp_string = std::to_string(num_topics);
            temp_string_vec.push_back(temp_string);
            topic_dict[num_topics] = 1;
            num_topics            += 1;
         }else{
            std::vector <std::string> temp_string_vec_2;
            for (int j = 0; j < int(temp_string_vec.size()); ++j) {
               for (int k = 0; k < initial_branch; ++k) {
                  std::string temp_string = temp_string_vec[j] + "," + std::to_string(num_topics);
                  temp_string_vec_2.push_back(temp_string);
                  topic_dict[num_topics] = 1;
                  num_topics            += 1;
               }
            }
            temp_string_vec.clear();
            for (int j = 0; j < int(temp_string_vec_2.size()); ++j) {
               temp_string_vec.push_back(temp_string_vec_2[j]);
            }
         }
      }

      // add to path_dict
      for (int i = 0; i < int(temp_string_vec.size()); ++i) {
         std::string temp_string = temp_string_vec[i];
         //std::cout << temp_string << std::endl;
         std::vector <int> v1 = Split2IntVector(temp_string);
         path_dict[i] = v1;
      }

      for (auto itr = path_dict.begin(); itr != path_dict.end(); ++itr) {
         std::vector <int> v1 = itr->second;
         std::string       temp_string;
         for (int i = 0; i < int(v1.size()); ++i) {
            temp_string += std::to_string(v1[i]) + ",";
         }
         std::cout << "Path id: " << itr->first << " " << temp_string << std::endl;
         num_path += 1;
      }
      std::cout << "num_topics " << num_topics << std::endl;
      std::cout << "num_path " << num_path << std::endl;
   }

   void create_initial_hierarchy_rev(const std::vector <int> branch_vec) {
      num_topics = 0;
      num_path   = 0;
      num_depth  = int(branch_vec.size());
      topic_dict.clear();
      path_dict.clear();

      std::vector <std::string> temp_string_vec;
      for (int i = 0; i < int(branch_vec.size()); ++i) {
         std::vector <std::string> temp_string_vec_0 = temp_string_vec;
         temp_string_vec.clear();
         if (int(temp_string_vec_0.size()) > 0) {
            for (int j = 0; j < int(temp_string_vec_0.size()); ++j) {
               for (int k = 0; k < branch_vec[i]; ++k) {
                  std::string temp_string = temp_string_vec_0[j] + "," + std::to_string(num_topics);
                  temp_string_vec.push_back(temp_string);
                  topic_dict[num_topics] = 1;
                  num_topics            += 1;
               }
            }
         }else{
            std::string temp_string = std::to_string(num_topics);
            temp_string_vec.push_back(temp_string);
            topic_dict[num_topics] = 1;
            num_topics            += 1;
         }
      }

      // add to path_dict
      for (int i = 0; i < int(temp_string_vec.size()); ++i) {
         std::string temp_string = temp_string_vec[i];
         //std::cout << temp_string << std::endl;
         std::vector <int> v1 = Split2IntVector(temp_string);
         path_dict[i] = v1;
      }

      for (auto itr = path_dict.begin(); itr != path_dict.end(); ++itr) {
         std::vector <int> v1 = itr->second;
         std::string       temp_string;
         for (int i = 0; i < int(v1.size()); ++i) {
            temp_string += std::to_string(v1[i]) + ",";
         }
         std::cout << "Path id: " << itr->first << " " << temp_string << std::endl;
         num_path += 1;
      }
      std::cout << "num_topics " << num_topics << std::endl;
      std::cout << "num_path " << num_path << std::endl;

      for (int i = 0; i < num_topics; ++i) {
         topic2num[i]   = 0.0;
         topic2mu[i]    = zero_mu;
         topic2Sigma[i] = zero_Sigma;
      }
   }

   void create_initial_assignment(const int version) {
      if (version == 0) {
         if (path_dict.find(0) == path_dict.end() || num_path < 1 || num_depth < 1) {
            std::cout << "Need to create hierarchy first" << std::endl;
         }else{
            path2num.clear();
            doc2path.clear();
            const gsl_rng_type *T;
            gsl_rng **          r;
            gsl_rng_env_setup();
            T = gsl_rng_mt19937;
            r = (gsl_rng **)malloc(100 * sizeof(gsl_rng *));
            std::random_device rand_device;
            std::mt19937       mt(rand_device());
            for (int i = 0; i < 1; i++) {
               r[i] = gsl_rng_alloc(T);
               gsl_rng_set(r[i], mt());
            }
            num_docs = 0;
            for (int i = 0; i < int(doc_word_topic.rows()); ++i) {
               int path = -1;
               if (doc2path.find(doc_word_topic(i, 0)) == doc2path.end()) {
                  path = int(floor(float(num_path) * gsl_rng_uniform(r[0])));
                  doc2path[doc_word_topic(i, 0)] = path;
                  std::vector <int> temp_vec;
                  temp_vec.push_back(i);
                  doc2place_vec[doc_word_topic(i, 0)] = temp_vec;
                  if (path2num.find(path) == path2num.end()) {
                     path2num[path] = 1.0;
                  }else{
                     path2num[path] += 1.0;
                  }
                  num_docs += 1;
               }else{
                  doc_word_topic(i, 2) = doc2path[doc_word_topic(i, 0)];
                  doc2place_vec[doc_word_topic(i, 0)].push_back(i);
               }
               doc_word_topic(i, 2) = doc2path[doc_word_topic(i, 0)];

               std::vector <int> new_path_vec = path_dict[doc_word_topic(i, 2)];
               int depth = -1;
               if (gsl_rng_uniform(r[0]) < mix_ratio) {
                  std::vector <float> level_ratio;
                  float agg_level = 0;
                  for (int i = 0; i < int(new_path_vec.size()); ++i) {
                     float first       = 1.0;
                     float first_nume  = hyper_m * hyper_pi + level_gamma;
                     float first_denom = hyper_pi;
                     for (int j = i; j < int(new_path_vec.size()); ++j) {
                        first_denom += level_gamma;
                     }
                     first = first_nume / first_denom;
                     float second = 1.0;
                     for (int j = 0; j < i; ++j) {
                        float second_nume_inside = 0.0;
                        for (int k = j + 1; k < int(new_path_vec.size()); ++k) {
                           second_nume_inside += level_gamma;
                        }
                        float second_nume         = (1.0 - hyper_m) * hyper_pi + second_nume_inside;
                        float second_denom_inside = 0.0;
                        for (int k = j; k < int(new_path_vec.size()); ++k) {
                           second_denom_inside += level_gamma;
                        }
                        float second_denom = hyper_pi + second_denom_inside;
                        second = second * (second_nume / second_denom);
                     }
                     float temp = first * second;
                     level_ratio.push_back(temp);
                     agg_level += temp;
                  }

                  float agg_topic = 0;
                  float max_val   = -9999999999999;
                  std::vector <float> ratio;
                  for (int j = 0; j < int(new_path_vec.size()); ++j) {
                     float log_prob_a = std::log(level_ratio[j]) - std::log(agg_level);
                     float log_prob   = log_prob_a;
                     ratio.push_back(log_prob);
                     if (max_val < ratio[j]) {
                        max_val = ratio[j];
                     }
                  }
                  agg_topic = 0;
                  for (int j = 0; j < int(ratio.size()); ++j) {
                     ratio[j]   = ratio[j] - max_val;
                     ratio[j]   = std::exp(ratio[j]);
                     agg_topic += ratio[j];
                  }
                  // sample
                  double prob[ratio.size()]; // Probability array
                  for (int j = 0; j < int(ratio.size()); ++j) {
                     prob[j] = double(ratio[j] / agg_topic);
                  }
                  unsigned int mult_op[int(ratio.size())];
                  int          num_sample = 1;
                  gsl_ran_multinomial(r[0], int(ratio.size()), num_sample, prob, mult_op);
                  int new_level = -1;
                  //int new_topic = -1;
                  for (int j = 0; j < int(ratio.size()); ++j) {
                     if (mult_op[j] == 1) {
                        new_level = j;
                        break;
                     }
                  }
                  depth = new_level;
               }else{
                  float div       = 1.0 / float(new_path_vec.size());
                  int   new_level = -1;
                  for (int k = 0; k < int(new_path_vec.size()); ++k) {
                     float thres = float(k + 1) * div + 0.000000001;
                     if (word2cdf[doc_word_topic(i, 1)] <= thres) {
                        new_level = k;
                        break;
                     }
                  }
                  depth = new_level;
               }
               int topic = path_level2topic(doc2path[doc_word_topic(i, 0)], depth);
               doc_word_topic(i, 3) = depth;
               doc_word_topic(i, 4) = topic;
            }
         }
      }else if (version == 1) {
         if (path_dict.find(0) == path_dict.end() || num_path < 1 || num_depth < 1) {
            std::cout << "Need to create hierarchy first" << std::endl;
         }else{
            path2num.clear();
            doc2path.clear();
            const gsl_rng_type *T;
            gsl_rng **          r;
            gsl_rng_env_setup();
            T = gsl_rng_mt19937;
            r = (gsl_rng **)malloc(100 * sizeof(gsl_rng *));
            std::random_device rand_device;
            std::mt19937       mt(rand_device());
            for (int i = 0; i < 1; i++) {
               r[i] = gsl_rng_alloc(T);
               gsl_rng_set(r[i], mt());
            }
            num_docs = 0;
            for (int i = 0; i < int(doc_word_topic.rows()); ++i) {
               int path = -1;
               if (doc2path.find(doc_word_topic(i, 0)) == doc2path.end()) {
                  path = int(floor(float(num_path) * gsl_rng_uniform(r[0])));
                  doc2path[doc_word_topic(i, 0)] = path;
                  std::vector <int> temp_vec;
                  temp_vec.push_back(i);
                  doc2place_vec[doc_word_topic(i, 0)] = temp_vec;
                  if (path2num.find(path) == path2num.end()) {
                     path2num[path] = 1.0;
                  }else{
                     path2num[path] += 1.0;
                  }
                  num_docs += 1;
               }else{
                  doc_word_topic(i, 2) = doc2path[doc_word_topic(i, 0)];
                  doc2place_vec[doc_word_topic(i, 0)].push_back(i);
               }
               doc_word_topic(i, 2) = doc2path[doc_word_topic(i, 0)];

               std::vector <int>   new_path_vec = path_dict[doc_word_topic(i, 2)];
               std::vector <float> level_ratio;
               float agg_level = 0;
               for (int i = 0; i < int(new_path_vec.size()); ++i) {
                  float first       = 1.0;
                  float first_nume  = hyper_m * hyper_pi + level_gamma;
                  float first_denom = hyper_pi;
                  for (int j = i; j < int(new_path_vec.size()); ++j) {
                     first_denom += level_gamma;
                  }
                  first = first_nume / first_denom;
                  float second = 1.0;
                  for (int j = 0; j < i; ++j) {
                     float second_nume_inside = 0.0;
                     for (int k = j + 1; k < int(new_path_vec.size()); ++k) {
                        second_nume_inside += level_gamma;
                     }
                     float second_nume         = (1.0 - hyper_m) * hyper_pi + second_nume_inside;
                     float second_denom_inside = 0.0;
                     for (int k = j; k < int(new_path_vec.size()); ++k) {
                        second_denom_inside += level_gamma;
                     }
                     float second_denom = hyper_pi + second_denom_inside;
                     second = second * (second_nume / second_denom);
                  }
                  float temp = first * second;
                  level_ratio.push_back(temp);
                  agg_level += temp;
               }

               float agg_topic = 0;
               float max_val   = -9999999999999;
               std::vector <float> ratio;
               for (int j = 0; j < int(new_path_vec.size()); ++j) {
                  float log_prob_a = std::log(level_ratio[j]) - std::log(agg_level);
                  float log_prob   = log_prob_a;
                  ratio.push_back(log_prob);
                  if (max_val < ratio[j]) {
                     max_val = ratio[j];
                  }
               }
               agg_topic = 0;
               for (int j = 0; j < int(ratio.size()); ++j) {
                  ratio[j]   = ratio[j] - max_val;
                  ratio[j]   = std::exp(ratio[j]);
                  agg_topic += ratio[j];
               }
               // sample
               double prob[ratio.size()]; // Probability array
               for (int j = 0; j < int(ratio.size()); ++j) {
                  prob[j] = double(ratio[j] / agg_topic);
               }
               unsigned int mult_op[int(ratio.size())];
               int          num_sample = 1;
               gsl_ran_multinomial(r[0], int(ratio.size()), num_sample, prob, mult_op);
               int new_level = -1;
               //int new_topic = -1;
               for (int j = 0; j < int(ratio.size()); ++j) {
                  if (mult_op[j] == 1) {
                     new_level = j;
                     break;
                  }
               }
               int depth = new_level;
               int topic = path_level2topic(doc2path[doc_word_topic(i, 0)], depth);
               doc_word_topic(i, 3) = depth;
               doc_word_topic(i, 4) = topic;
            }
         }
      }else if (version == 2) {
         if (path_dict.find(0) == path_dict.end() || num_path < 1 || num_depth < 1) {
            std::cout << "Need to create hierarchy first" << std::endl;
         }else{
            path2num.clear();
            doc2path.clear();
            const gsl_rng_type *T;
            gsl_rng **          r;
            gsl_rng_env_setup();
            T = gsl_rng_mt19937;
            r = (gsl_rng **)malloc(100 * sizeof(gsl_rng *));
            std::random_device rand_device;
            std::mt19937       mt(rand_device());
            for (int i = 0; i < 1; i++) {
               r[i] = gsl_rng_alloc(T);
               gsl_rng_set(r[i], mt());
            }
            num_docs = 0;
            for (int i = 0; i < int(doc_word_topic.rows()); ++i) {
               int path = -1;
               if (doc2path.find(doc_word_topic(i, 0)) == doc2path.end()) {
                  path = int(floor(float(num_path) * gsl_rng_uniform(r[0])));
                  doc2path[doc_word_topic(i, 0)] = path;
                  std::vector <int> temp_vec;
                  temp_vec.push_back(i);
                  doc2place_vec[doc_word_topic(i, 0)] = temp_vec;
                  if (path2num.find(path) == path2num.end()) {
                     path2num[path] = 1.0;
                  }else{
                     path2num[path] += 1.0;
                  }
                  num_docs += 1;
               }else{
                  doc_word_topic(i, 2) = doc2path[doc_word_topic(i, 0)];
                  doc2place_vec[doc_word_topic(i, 0)].push_back(i);
               }
               int depth = int(floor(float(num_depth) * gsl_rng_uniform(r[0])));
               int topic = path_level2topic(doc2path[doc_word_topic(i, 0)], depth);
               doc_word_topic(i, 2) = doc2path[doc_word_topic(i, 0)];
               doc_word_topic(i, 3) = depth;
               doc_word_topic(i, 4) = topic;
            }
         }
      }else{
         doc2path.clear();
         num_docs = 0;
         for (int i = 0; i < int(doc_word_topic.rows()); ++i) {
            if (doc2path.find(doc_word_topic(i, 0)) == doc2path.end()) {
               int path = doc_word_topic(i, 5);
               doc2path[doc_word_topic(i, 0)] = path;
               doc_word_topic(i, 2)           = path;
               std::vector <int> temp_vec;
               temp_vec.push_back(i);
               doc2place_vec[doc_word_topic(i, 0)] = temp_vec;
               if (path2num.find(path) == path2num.end()) {
                  path2num[path] = 1.0;
               }else{
                  path2num[path] += 1.0;
               }
               num_docs += 1;
            }else{
               doc_word_topic(i, 2) = doc2path[doc_word_topic(i, 0)];
               doc2place_vec[doc_word_topic(i, 0)].push_back(i);
            }
            // depth topic
            doc_word_topic(i, 3) = doc_word_topic(i, 6);
            doc_word_topic(i, 4) = doc_word_topic(i, 7);
         }
      }
   }

   float ghlda_posterior_predictive_robust(const int& doc,
                                           const std::vector <int>& topic_vec,
                                           const int& level,
                                           const float& nu, const float& kappa,
                                           std::unordered_map <int, float>& topic2num_temp,
                                           std::unordered_map <int, Eigen::MatrixXf>& topic2mu_temp,
                                           std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_temp,
                                           std::unordered_map <int, float> topic2third_nume_inside,
                                           const int& log_true, const int& verbose) {
      Eigen::MatrixXf Psi_t = level2Psi[level];

      std::vector <int> place_vec = doc2place_vec[doc];
      std::vector <int> place_level_vec;
      for (int i = 0; i < int(place_vec.size()); ++i) {
         if (doc_word_topic(place_vec[i], 3) == level) {
            place_level_vec.push_back(place_vec[i]);
         }
      }

      const int topic = topic_vec[level];
      // KOKOGA MONDAI
      float n = 0;
      float m = 0;
      n = float(topic2num_temp[topic]);   // - float(place_vec.size());
      m = float(place_level_vec.size());

      // Actually we have to re calculate mu and sigma
      float log_prob           = 0.0;
      float prob               = 0.0;
      float first              = 0.0;
      float second_nume        = 0.0;
      float second_denom       = 0.0;
      float third_nume_inside  = 0.0;
      float third_denom_inside = 0.0;
      float third_nume         = 0.0;
      float third_denom        = 0.0;
      float fourth_nume        = 0.0;
      float fourth_denom       = 0.0;
      // Choleskey fast
      std::string method = "llt";

      first = 0 - std::log(float(place_vec.size()) * embedding_dimension / 2.0 * std::log(M_PI));
      if (verbose == 1) {
         std::cout << "  first " << first << std::endl;
      }

      for (int d = 0; d < embedding_dimension; ++d) {
         second_nume  = second_nume + std::lgamma((nu + n + m - float(d)) / 2.0);
         second_denom = second_denom + std::lgamma((nu + n - float(d)) / 2.0);
      }

      if (verbose == 1) {
         std::cout << "  second_nume " << second_nume << std::endl;
         std::cout << "  second_denom " << second_denom << std::endl;
      }

      std::unordered_map <int, float> thres2third_nume_inside;

      if (1 == 1) {
         //if(topic2third_nume_inside.find(topic)==topic2third_nume_inside.end()){

         Eigen::MatrixXf mu           = topic2mu_temp[topic];
         Eigen::MatrixXf Sigma        = topic2Sigma_temp[topic];
         Eigen::MatrixXf Sigma_stable = Psi_t + Sigma + ((kappa * n) / (kappa + n)) * (mu - embedding_center).transpose() * (mu - embedding_center);
         Sigma_stable = rescale * Sigma_stable;

         if (method == "llt") {
            Eigen::LLT <Eigen::MatrixXf> llt;
            llt.compute(Sigma_stable);
            auto& U = llt.matrixL();
            for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
               if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
                  third_nume_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
               }else{
                  third_nume_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
               }
               thres2third_nume_inside[i + 1] = third_nume_inside;
            }
         }        //if(method == "llt"){
         if (verbose == 1) {
            std::cout << "third_nume_inside " << third_nume_inside << std::endl;
         }
         topic2third_nume_inside[topic] = third_nume_inside;
      }else{
         third_nume_inside = topic2third_nume_inside[topic];
      }

      // third_denom
      float           N_temp     = topic2num_temp[topic];
      Eigen::MatrixXf mu_temp    = topic2mu_temp[topic];
      Eigen::MatrixXf Sigma_temp = topic2Sigma_temp[topic];

      int hantei = 0;
      if (N_temp == 0) {
         hantei = 1;
      }

      for (int i = 0; i < int(place_level_vec.size()); ++i) {
         int             alter_word = doc_word_topic(place_level_vec[i], 1);
         Eigen::MatrixXf alter_x    = embedding_matrix.row(alter_word);
         if (hantei == 0) {
            ghlda_update_mu_sigma(topic, alter_x, N_temp, mu_temp, Sigma_temp);
         }else{
            N_temp += 1;
            mu_temp = mu_temp + alter_x;
         }
      }

      if (hantei == 1 && N_temp > 0) {
         mu_temp = mu_temp / N_temp;
      }

      float value  = (kappa * n + m);
      float value2 = (kappa + n + m);
      float value3 = ((kappa * n + m) / (kappa + n + m));

      Eigen::MatrixXf Sigma_stable = Psi_t + Sigma_temp + ((kappa * n + m) / (kappa + n + m)) * (mu_temp - embedding_center).transpose() * (mu_temp - embedding_center);
      Sigma_stable = rescale * Sigma_stable;
      if (method == "llt") {
         Eigen::LLT <Eigen::MatrixXf> llt;
         llt.compute(Sigma_stable);
         auto& U = llt.matrixL();
         for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
            if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
               third_denom_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
            }else{
               third_denom_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
            }
         }
         third_nume  = ((nu + n) / 2.0) * third_nume_inside;
         third_denom = ((nu + m + n) / 2.0) * third_denom_inside;
      }   //if(method == "llt"){

      if (verbose == 1) {
         //std::cout << "  hantei " << hantei << std::endl;
         //std::cout << "  int(place_level_vec.size()) " << int(place_level_vec.size()) << std::endl;
         //std::cout << "  mu_temp" << std::endl;
         //std::cout << mu_temp << std::endl;
         //std::cout << "  Sigma_temp " << std::endl;
         //std::cout << Sigma_temp << std::endl;
         //std::cout << "  Psi " << std::endl;
         //std::cout << Psi << std::endl;
         //std::cout << "  Sigma_stable " << std::endl;
         //std::cout << Sigma_stable << std::endl;
         //std::cout << "  embedding_center " << std::endl;
         //std::cout << embedding_center << std::endl;
         std::cout << "  value " << value << std::endl;
         std::cout << "  value2 " << value2 << std::endl;
         std::cout << "  value3 " << value3 << std::endl;
         std::cout << "  third_nume " << third_nume << std::endl;
         std::cout << "  third_denom " << third_denom << std::endl;
      }

      fourth_nume  = (embedding_dimension / 2.0) * std::log(kappa + n);
      fourth_denom = (embedding_dimension / 2.0) * std::log(kappa + n + m);

      if (verbose == 1) {
         std::cout << "  n " << n << std::endl;
         std::cout << "  m " << m << std::endl;
         std::cout << "  fourth_nume " << fourth_nume << std::endl;
         std::cout << "  fourth_denom " << fourth_denom << std::endl;
      }

      log_prob = first + second_nume - second_denom + third_nume - third_denom + fourth_nume - fourth_denom;
      log_prob = log_prob - inner_rescale;

      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      return(prob);
   }

   float ghlda_posterior_predictive_single_robust(const int& place,
                                                  const int& level,
                                                  const int& topic,
                                                  const float& nu, const float& kappa,
                                                  std::unordered_map <int, float>& topic2num_temp,
                                                  std::unordered_map <int, Eigen::MatrixXf>& topic2mu_temp,
                                                  std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_temp,
                                                  const int& log_true, const int& verbose) {
      Eigen::MatrixXf Psi_t = level2Psi[level];

      //std::vector<int> place_vec = doc2place_vec[doc];
      std::vector <int> place_level_vec;
      place_level_vec.push_back(place);

      // KOKOGA MONDAI
      float n = 0;
      float m = 0;
      n = float(topic2num_temp[topic]);   // - float(place_vec.size());
      m = float(place_level_vec.size());

      // Actually we have to re calculate mu and sigma
      float log_prob           = 0.0;
      float prob               = 0.0;
      float first              = 0.0;
      float second_nume        = 0.0;
      float second_denom       = 0.0;
      float third_nume_inside  = 0.0;
      float third_denom_inside = 0.0;
      float third_nume         = 0.0;
      float third_denom        = 0.0;
      float fourth_nume        = 0.0;
      float fourth_denom       = 0.0;
      // Choleskey fast
      std::string method = "llt";

      // KOKO
      first = 0 - std::log(float(place_level_vec.size()) * embedding_dimension / 2.0 * std::log(M_PI));
      if (verbose == 1) {
         std::cout << "  first " << first << std::endl;
      }

      for (int d = 0; d < embedding_dimension; ++d) {
         second_nume  = second_nume + std::lgamma((nu + n + m - float(d)) / 2.0);
         second_denom = second_denom + std::lgamma((nu + n - float(d)) / 2.0);
      }

      //second_nume = std::lgamma((nu + n + m + float(embedding_dimension)) / 2.0);
      //second_denom = std::lgamma((nu + n + float(embedding_dimension)) / 2.0);

      if (verbose == 1) {
         std::cout << "  second_nume " << second_nume << std::endl;
         std::cout << "  second_denom " << second_denom << std::endl;
      }

      std::unordered_map <int, float> thres2third_nume_inside;

      if (1 == 1) {
         //if(topic2third_nume_inside.find(topic)==topic2third_nume_inside.end()){
         Eigen::MatrixXf mu           = topic2mu_temp[topic];
         Eigen::MatrixXf Sigma        = topic2Sigma_temp[topic];
         Eigen::MatrixXf Sigma_stable = Psi_t + Sigma + ((kappa * n) / (kappa + n)) * (mu - embedding_center).transpose() * (mu - embedding_center);
         Sigma_stable = rescale * Sigma_stable;
         if (method == "llt") {
            Eigen::LLT <Eigen::MatrixXf> llt;
            llt.compute(Sigma_stable);
            auto& U = llt.matrixL();
            for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
               if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
                  third_nume_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
               }else{
                  third_nume_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
               }
               thres2third_nume_inside[i + 1] = third_nume_inside;
            }
         }        //if(method == "llt"){
         if (verbose == 1) {
            std::cout << "third_nume_inside " << third_nume_inside << std::endl;
         }
      }else{
      }

      //third_nume = ((nu + n) / 2.0) * third_nume_inside;

      // third_denom
      float           N_temp     = topic2num_temp[topic];
      Eigen::MatrixXf mu_temp    = topic2mu_temp[topic];
      Eigen::MatrixXf Sigma_temp = topic2Sigma_temp[topic];

      int hantei = 0;
      if (N_temp == 0) {
         hantei = 1;
      }

      for (int i = 0; i < int(place_level_vec.size()); ++i) {
         int             alter_word = doc_word_topic(place_level_vec[i], 1);
         Eigen::MatrixXf alter_x    = embedding_matrix.row(alter_word);
         if (hantei == 0) {
            ghlda_update_mu_sigma(topic, alter_x, N_temp, mu_temp, Sigma_temp);
         }else{
            N_temp += 1;
            mu_temp = mu_temp + alter_x;
         }
      }

      if (hantei == 1 && N_temp > 0) {
         mu_temp = mu_temp / N_temp;
      }

      float value  = (kappa * n + m);
      float value2 = (kappa + n + m);
      float value3 = ((kappa * n + m) / (kappa + n + m));

      Eigen::MatrixXf Sigma_stable = Psi_t + Sigma_temp + ((kappa * n + m) / (kappa + n + m)) * (mu_temp - embedding_center).transpose() * (mu_temp - embedding_center);
      Sigma_stable = rescale * Sigma_stable;
      if (method == "llt") {
         Eigen::LLT <Eigen::MatrixXf> llt;
         llt.compute(Sigma_stable);
         auto& U = llt.matrixL();
         for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
            if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
               third_denom_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
            }else{
               third_denom_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
            }
         }
         third_nume  = ((nu + n) / 2.0) * third_nume_inside;
         third_denom = ((nu + m + n) / 2.0) * third_denom_inside;
      }

      if (verbose == 1) {
         std::cout << "  value " << value << std::endl;
         std::cout << "  value2 " << value2 << std::endl;
         std::cout << "  value3 " << value3 << std::endl;
         std::cout << "  third_nume " << third_nume << std::endl;
         std::cout << "  third_denom " << third_denom << std::endl;
      }

      fourth_nume  = (embedding_dimension / 2.0) * std::log(kappa + n);
      fourth_denom = (embedding_dimension / 2.0) * std::log(kappa + n + m);

      if (verbose == 1) {
         std::cout << "  n " << n << std::endl;
         std::cout << "  m " << m << std::endl;
         std::cout << "  fourth_nume " << fourth_nume << std::endl;
         std::cout << "  fourth_denom " << fourth_denom << std::endl;
      }

      log_prob = first + second_nume - second_denom + third_nume - third_denom + fourth_nume - fourth_denom;
      log_prob = log_prob - inner_rescale;
      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      return(prob);
   }

   void ghlda_calc_tables_parameters_from_assign() {
      if (embedding_dimension == 0) {
         std::cout << "Please set embedding_dimension" << std::endl;
      }else{
         doc2level_vec.clear();
         doc2path.clear();
         path2num.clear();
         word2count.clear();
         word2topic_vec.clear();
         for (auto itr = topic2num.begin(); itr != topic2num.end(); ++itr) {
            topic2num[itr->first] = 0.0;
         }
         for (auto itr = topic2mu.begin(); itr != topic2mu.end(); ++itr) {
            topic2mu[itr->first] = zero_mu;
         }
         for (auto itr = topic2Sigma.begin(); itr != topic2Sigma.end(); ++itr) {
            topic2Sigma[itr->first] = zero_Sigma;
         }

         topic2num[-1]   = 0.0;
         topic2mu[-1]    = zero_mu;
         topic2Sigma[-1] = zero_Sigma;

         //std::cout << "ghlda_calc_tables After Initialization" << std::endl;

         for (int i = 0; i < doc_word_topic.rows(); ++i) {
            int doc   = doc_word_topic(i, 0);
            int word  = doc_word_topic(i, 1);
            int path  = doc_word_topic(i, 2);
            int level = doc_word_topic(i, 3);
            int topic = doc_word_topic(i, 4);

            Eigen::MatrixXf x = embedding_matrix.row(word);
            // Create tables
            if (doc2level_vec.find(doc) == doc2level_vec.end()) {
               doc2level_vec[doc] = level_gamma * Eigen::MatrixXf::Ones(num_depth + 1000, 1);
            }
            doc2level_vec[doc](level, 0) += 1.0;
            if (doc2path.find(doc) == doc2path.end()) {
               doc2path[doc] = path;
               if (path2num.find(path) == path2num.end()) {
                  path2num[path] = 1.0;
               }else{
                  path2num[path] += 1.0;
               }
            }
            if (word2count.find(word) == word2count.end()) {
               word2count[word] = 1.0;
            }else{
               word2count[word] += 1.0;
            }
            if (word2topic_vec.find(word) == word2topic_vec.end()) {
               word2topic_vec[word] = Eigen::MatrixXf::Zero(num_topics + 1000, 1);
            }
            num_not_assigned = 0.0;
            if (topic != -1) {
               word2topic_vec[word](topic, 0) += 1.0;
            }else{
               num_not_assigned += 1.0;
            }

            if (topic != -1) {
               // Create Parameters
               if (topic2num.find(topic) == topic2num.end()) {
                  topic2num[topic] = 1.0;
                  topic2mu[topic]  = x;
               }else{
                  topic2num[topic] += 1.0;
                  topic2mu[topic]  += x;
               }
            }else{
            }
         }
         //std::cout << "ghlda_calc_tables After topic2mu" << std::endl;

         // Normalize
         for (auto itr = topic2num.begin(); itr != topic2num.end(); ++itr) {
            if (itr->second > 0) {
               topic2mu[itr->first] = topic2mu[itr->first] / itr->second;
            }
         }
         //std::cout << "ghlda_calc_tables After Normalize" << std::endl;

         // Variance
         for (int i = 0; i < doc_word_topic.rows(); ++i) {
            int             word  = doc_word_topic(i, 1);
            int             topic = doc_word_topic(i, 4);
            Eigen::MatrixXf x     = embedding_matrix.row(word);
            if (topic != -1) {
               if (topic2Sigma.find(topic) == topic2Sigma.end()) {
                  Eigen::MatrixXf outer_matrix = (x - topic2mu[topic]).transpose() * (x - topic2mu[topic]);
                  topic2Sigma[topic] = outer_matrix;
               }else{
                  Eigen::MatrixXf outer_matrix = (x - topic2mu[topic]).transpose() * (x - topic2mu[topic]);
                  topic2Sigma[topic] += outer_matrix;
               }
            }
         }
         //std::cout << "ghlda_calc_tables After Variance" << std::endl;
      }
   }

   void ghlda_update_parameters_subtract(const int topic, Eigen::MatrixXf& x) {
      // Subtract parameters
      float N0 = topic2num[topic];

      if (N0 < 1.1) {
         topic2mu[topic]    = zero_mu;
         topic2Sigma[topic] = zero_Sigma;
         topic2num[topic]   = 0.0;
      }else{
         Eigen::MatrixXf mu0       = topic2mu[topic];
         Eigen::MatrixXf Sigma0    = topic2Sigma[topic];
         Eigen::MatrixXf mu0_minus = (N0 * mu0 - x) / (N0 - 1.0);
         // (4*Sigma2 + 4*(m1-m2).transpose()*(m1-m2) - (x4-m1).transpose()*(x4-m1)) /3
         Eigen::MatrixXf Sigma0_minus = Sigma0 + N0 * (mu0_minus - mu0).transpose() * (mu0_minus - mu0) - (x - mu0_minus).transpose() * (x - mu0_minus);
         topic2num[topic]   = N0 - 1.0;
         topic2mu[topic]    = mu0_minus;
         topic2Sigma[topic] = Sigma0_minus;
      }
   }

   void ghlda_update_parameters_subtract_thread(const int topic, const Eigen::MatrixXf& x,
                                                std::unordered_map <int, float>& topic2num_temp,
                                                std::unordered_map <int, Eigen::MatrixXf>& topic2mu_temp,
                                                std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_temp) {
      // Subtract parameters
      float N0 = topic2num_temp[topic];

      if (N0 < 1.1) {
         topic2mu_temp[topic]    = zero_mu;
         topic2Sigma_temp[topic] = zero_Sigma;
         topic2num_temp[topic]   = 0.0;
      }else{
         Eigen::MatrixXf mu0       = topic2mu_temp[topic];
         Eigen::MatrixXf Sigma0    = topic2Sigma_temp[topic];
         Eigen::MatrixXf mu0_minus = (N0 * mu0 - x) / (N0 - 1.0);
         // (4*Sigma2 + 4*(m1-m2).transpose()*(m1-m2) - (x4-m1).transpose()*(x4-m1)) /3
         Eigen::MatrixXf Sigma0_minus = Sigma0 + N0 * (mu0_minus - mu0).transpose() * (mu0_minus - mu0) - (x - mu0_minus).transpose() * (x - mu0_minus);
         topic2mu_temp[topic]    = mu0_minus;
         topic2Sigma_temp[topic] = Sigma0_minus;
         topic2num_temp[topic]   = N0 - 1.0;
      }
   }

   void ghlda_update_parameters_add(const int& new_topic, const Eigen::MatrixXf& x) {
      if (topic2num.find(new_topic) != topic2num.end()) {
         // Update parameters
         float           N3       = topic2num[new_topic];
         Eigen::MatrixXf mu3      = topic2mu[new_topic];
         Eigen::MatrixXf Sigma3   = topic2Sigma[new_topic];
         Eigen::MatrixXf mu3_plus = (N3 * mu3 + x) / (N3 + 1.0);
         // # (3*Sigma1 + 3*(m1-m2).transpose()*(m1-m2) + (x4-m2).transpose()*(x4-m2)) / 4
         Eigen::MatrixXf Sigma3_plus = Sigma3 + N3 * (mu3 - mu3_plus).transpose() * (mu3 - mu3_plus) + (x - mu3_plus).transpose() * (x - mu3_plus);
         topic2num[new_topic]   = N3 + 1.0;
         topic2mu[new_topic]    = mu3_plus;
         topic2Sigma[new_topic] = Sigma3_plus;
      }else{
         // KOKOKA?
         topic2num[new_topic]   = 1.0;
         topic2mu[new_topic]    = x;
         topic2Sigma[new_topic] = zero_Sigma;
      }
   }

   void ghlda_update_parameters_add_thread(const int new_topic, const Eigen::MatrixXf& x,
                                           std::unordered_map <int, float>& topic2num_temp,
                                           std::unordered_map <int, Eigen::MatrixXf>& topic2mu_temp,
                                           std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_temp) {
      if (topic2num_temp.find(new_topic) != topic2num_temp.end()) {
         // Update parameters
         float           N3       = topic2num_temp[new_topic];
         Eigen::MatrixXf mu3      = topic2mu_temp[new_topic];
         Eigen::MatrixXf Sigma3   = topic2Sigma_temp[new_topic];
         Eigen::MatrixXf mu3_plus = (N3 * mu3 + x) / (N3 + 1.0);
         // # (3*Sigma1 + 3*(m1-m2).transpose()*(m1-m2) + (x4-m2).transpose()*(x4-m2)) / 4
         Eigen::MatrixXf Sigma3_plus = Sigma3 + N3 * (mu3 - mu3_plus).transpose() * (mu3 - mu3_plus) + (x - mu3_plus).transpose() * (x - mu3_plus);
         topic2num_temp[new_topic]   = N3 + 1.0;
         topic2mu_temp[new_topic]    = mu3_plus;
         topic2Sigma_temp[new_topic] = Sigma3_plus;
      }else{
         // KOKOKA?
         topic2num_temp[new_topic]   = 1.0;
         topic2mu_temp[new_topic]    = x;
         topic2Sigma_temp[new_topic] = zero_Sigma;
      }
   }

   void ghlda_collapsed_gibbs_sample_parallel(const int num_iteration, const int parallel_loop, int num_threads, const int verbose) {
      if (verbose == 1) {
         std::cout << "Enter ghlda_collapsed_gibbs_sample_parallel" << std::endl;
      }

      stop_hantei = 0;
      if (num_threads < 1) {
         num_threads = omp_get_max_threads();
         std::cout << "Using " << num_threads << " threads" << std::endl;
      }

      omp_set_num_threads(num_threads);

      const gsl_rng_type *T;
      gsl_rng **          r;
      gsl_rng_env_setup();
      T = gsl_rng_mt19937;
      r = (gsl_rng **)malloc(num_threads * sizeof(gsl_rng *));
      std::random_device rand_device;
      std::mt19937       mt(rand_device());
      for (int i = 0; i < num_threads; i++) {
         r[i] = gsl_rng_alloc(T);
         gsl_rng_set(r[i], mt());
      }
      if (verbose == 1) {
         std::cout << "After gsl_rng_set" << std::endl;
      }

      for (int itr_outer = 0; itr_outer < num_iteration; ++itr_outer) {
         if (stop_hantei == 1) {
            std::cout << "ERROR: COULD NOT SAMPLE TOPIC" << std::endl;
            break;
         }

         #pragma omp parallel for
         for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
            if (stop_hantei != 1) {
               int thread_id   = omp_get_thread_num();
               int doc_thread  = int(floor(float(num_docs) * gsl_rng_uniform(r[thread_id])));
               int path_thread = doc2path[doc_thread];
               if (verbose == 1) {
                  std::cout << "After path_thread" << std::endl;
               }

               //Used in path sampling
               std::unordered_map <int, float>           path2num_thread    = path2num;
               std::unordered_map <int, float>           topic2num_thread   = topic2num;
               std::unordered_map <int, Eigen::MatrixXf> topic2mu_thread    = topic2mu;
               std::unordered_map <int, Eigen::MatrixXf> topic2Sigma_thread = topic2Sigma;
               //Used in level sampling
               std::unordered_map <int, float>           topic2num_thread_2   = topic2num;
               std::unordered_map <int, Eigen::MatrixXf> topic2mu_thread_2    = topic2mu;
               std::unordered_map <int, Eigen::MatrixXf> topic2Sigma_thread_2 = topic2Sigma;

               if (1 == 1) {
                  //if(path2num_thread[path_thread]>0){
                  // Subtract Path
                  path2num_thread[path_thread] = path2num_thread[path_thread] - 1.0;

                  // Minus document assignment
                  std::vector <int> place_vec_thread = doc2place_vec[doc_thread];
                  int true_path = -1;

                  for (int i = 0; i < int(place_vec_thread.size()); ++i) {
                     int             alter_word  = doc_word_topic(place_vec_thread[i], 1);
                     Eigen::MatrixXf alter_x     = embedding_matrix.row(alter_word);
                     int             alter_topic = doc_word_topic(place_vec_thread[i], 4);
                     ghlda_update_parameters_subtract_thread(alter_topic, alter_x, topic2num_thread, topic2mu_thread, topic2Sigma_thread);
                  }// for(int i=0;i<int(place_vec_thread.size());++i){

                  if (verbose == 1) {
                     std::cout << "After ghlda_update_parameters_subtract_thread" << std::endl;
                  }

                  std::map <int, std::vector <int> > new_path_dict = path_dict;

                  if (verbose == 1) {
                     for (auto itr = new_path_dict.begin(); itr != new_path_dict.end(); ++itr) {
                        std::vector <int> v1          = itr->second;
                        std::string       temp_string = std::to_string(itr->first) + ":";
                        for (int i = 0; i < int(v1.size()); ++i) {
                           temp_string += std::to_string(v1[i]) + ",";
                        }
                        std::cout << temp_string << std::endl;
                     }
                  }//if(verbose == 1){

                  // Step A: sample path for a document
                  std::string                     prob_string = "";
                  float                           agg_topic   = 0;
                  float                           max_val     = -9999999999999;
                  std::vector <int>               path_id_vec;
                  std::vector <float>             ratio;
                  std::vector <std::string>       topic_string_vec;
                  std::unordered_map <int, float> topic2third_nume_inside;
                  for (auto itr = new_path_dict.begin(); itr != new_path_dict.end(); ++itr) {
                     int path_id = itr->first;
                     path_id_vec.push_back(path_id);
                     std::vector <int> topic_vec = itr->second;
                     float             temp      = 0.0;
                     float             temp_in   = 0.0;
                     prob_string = "";
                     for (int j = 0; j < int(topic_vec.size()); ++j) {
                        temp_in = ghlda_posterior_predictive_robust(doc_thread,
                                                                    topic_vec, j, nu, kappa, topic2num_thread, topic2mu_thread,
                                                                    topic2Sigma_thread, topic2third_nume_inside, 1, 0);
                        temp += temp_in;
                        float temp_val = topic2num_thread[topic_vec[j]];
                        prob_string = prob_string + "," + std::to_string(topic_vec[j]) + "-" + std::to_string(temp_val);
                     }
                     topic_string_vec.push_back(prob_string);
                     if (path2num_thread.find(path_id) != path2num_thread.end()) {
                        temp += std::log(path2num_thread[path_id] + path_gamma);
                     }else{
                        temp += std::log(path_gamma);
                     }
                     ratio.push_back(temp);
                     if (max_val < temp) {
                        max_val = temp;
                     }
                  }

                  if (verbose == 1) {
                     std::cout << "After Step A before sample" << std::endl;
                  }

                  // sample
                  prob_string = "";
                  for (int i = 0; i < int(ratio.size()); ++i) {
                     ratio[i]    = ratio[i] - max_val;
                     prob_string = prob_string + ":" + std::to_string(i) + "-" + topic_string_vec[i] + "," + std::to_string(ratio[i]) + "\n";
                     ratio[i]    = std::exp(ratio[i]);
                     agg_topic  += ratio[i];
                  }
                  double prob[int(ratio.size())]; // Probability array
                  for (int i = 0; i < int(ratio.size()); ++i) {
                     prob[i] = double(ratio[i] / agg_topic);
                  }
                  unsigned int mult_op[ratio.size()];
                  int          num_sample = 1;
                  gsl_ran_multinomial(r[thread_id], int(ratio.size()), num_sample, prob, mult_op);
                  int new_path = -1;
                  for (int i = 0; i < int(ratio.size()); ++i) {
                     if (mult_op[i] == 1) {
                        new_path = path_id_vec[i];
                        break;
                     }
                  }

                  if (verbose == 1) {
                     std::cout << "doc num " << doc_thread << std::endl;
                     std::cout << "True " << true_path << " Sample " << new_path << std::endl;
                     std::cout << prob_string << std::endl;
                  }
                  if (new_path < 0) {
                     #pragma omp critical
                     {
                        stop_hantei = 1;
                        std::cout << "ERROR STOP PATH" << std::endl;
                     }
                  }// if(new_path < 0){

                  // new_path
                  std::vector <int> new_path_vec = new_path_dict[new_path];

                  // Step B: sample level
                  if (stop_hantei != 1) {
                     // COPY
                     std::unordered_map <long long, Eigen::MatrixXf> doc2level_vec_thread = doc2level_vec;

                     // Create place_vec_thread shuffled
                     std::map <int, std::vector <int> > new_place_path_level;
                     std::random_device seed_gen;
                     std::mt19937       engine(seed_gen());
                     std::shuffle(place_vec_thread.begin(), place_vec_thread.end(), engine);
                     int inner_count = 0;

                     // for each word position
                     for (int itr_place = 0; itr_place < int(place_vec_thread.size()); ++itr_place) {
                        int             place     = place_vec_thread[itr_place];
                        int             word      = doc_word_topic(place, 1);
                        Eigen::MatrixXf x         = embedding_matrix.row(word);
                        int             old_level = doc_word_topic(place, 3);
                        int             old_topic = doc_word_topic(place, 4);

                        //SUBTRACT
                        ghlda_update_parameters_subtract_thread(old_topic, x,
                                                                topic2num_thread_2, topic2mu_thread_2, topic2Sigma_thread_2);

                        int hantei_update_doc2level = 0;
                        if (doc2level_vec_thread[doc_thread](old_level, 0) > 0) {
                           doc2level_vec_thread[doc_thread](old_level, 0) = doc2level_vec_thread[doc_thread](old_level, 0) - 1.0;
                           hantei_update_doc2level = 1;
                        }

                        // Create level topic ratio
                        std::vector <float> level_ratio;
                        float agg_level = 0.0;
                        if (level_allocation_type == 0) {// Pure LDA
                           for (int i = 0; i < int(new_path_vec.size()); ++i) {
                              float temp = alpha_level_vec(i) + doc2level_vec_thread[doc_thread](i, 0);
                              level_ratio.push_back(temp);
                              agg_level += temp;
                           }
                        }else{// HLDA
                           for (int i = 0; i < int(new_path_vec.size()); ++i) {
                              float first       = 1.0;
                              float first_nume  = hyper_m * hyper_pi + doc2level_vec_thread[doc_thread](i, 0);
                              float first_denom = hyper_pi;
                              for (int j = i; j < int(new_path_vec.size()); ++j) {
                                 first_denom += doc2level_vec_thread[doc_thread](j, 0);
                              }
                              first = first_nume / first_denom;
                              float second = 1.0;
                              for (int j = 0; j < i; ++j) {
                                 float second_nume_inside = 0.0;
                                 for (int k = j + 1; k < int(new_path_vec.size()); ++k) {
                                    second_nume_inside += doc2level_vec_thread[doc_thread](k, 0);
                                 }
                                 float second_nume         = (1.0 - hyper_m) * hyper_pi + second_nume_inside;
                                 float second_denom_inside = 0.0;
                                 for (int k = j; k < int(new_path_vec.size()); ++k) {
                                    second_denom_inside += doc2level_vec_thread[doc_thread](k, 0);
                                 }
                                 float second_denom = hyper_pi + second_denom_inside;
                                 second = second * (second_nume / second_denom);
                              }
                              float temp = first * second;
                              level_ratio.push_back(temp);
                              agg_level += temp;
                           }
                        }//if(level_allocation_type == 0){}else{}

                        float agg_topic = 0;
                        max_val = -9999999999999;
                        std::vector <float> ratio;
                        for (int j = 0; j < int(new_path_vec.size()); ++j) {
                           int   topic      = new_path_vec[j];
                           float log_prob_a = std::log(level_ratio[j]) - std::log(agg_level);
                           float log_prob_b = ghlda_posterior_predictive_single_robust(place, j, topic, nu, kappa,
                                                                                       topic2num_thread_2, topic2mu_thread_2,
                                                                                       topic2Sigma_thread_2, 1, verbose);
                           float log_prob = log_prob_a + log_prob_b;
                           ratio.push_back(log_prob);
                           if (max_val < ratio[j]) {
                              max_val = ratio[j];
                           }
                        }
                        agg_topic = 0;
                        for (int j = 0; j < int(ratio.size()); ++j) {
                           ratio[j]   = ratio[j] - max_val;
                           ratio[j]   = std::exp(ratio[j]);
                           agg_topic += ratio[j];
                        }
                        // sample
                        double prob[ratio.size()]; // Probability array
                        for (int j = 0; j < int(ratio.size()); ++j) {
                           prob[j] = double(ratio[j] / agg_topic);
                        }
                        unsigned int mult_op[int(ratio.size())];
                        int          num_sample = 1;
                        gsl_ran_multinomial(r[thread_id], int(ratio.size()), num_sample, prob, mult_op);
                        int new_level = -1;
                        int new_topic = -1;
                        for (int j = 0; j < int(ratio.size()); ++j) {
                           if (mult_op[j] == 1) {
                              new_level = j;
                              new_topic = new_path_vec[new_level];
                              break;
                           }
                        }

                        // ADD
                        ghlda_update_parameters_add_thread(new_topic, x, topic2num_thread_2, topic2mu_thread_2, topic2Sigma_thread_2);

                        if (hantei_update_doc2level == 1) {
                           doc2level_vec_thread[doc_thread](new_level, 0) = doc2level_vec_thread[doc_thread](new_level, 0) + 1.0;
                        }

                        std::vector <int> result;
                        result.push_back(place);
                        result.push_back(new_path);
                        result.push_back(new_level);
                        result.push_back(new_topic);
                        new_place_path_level[inner_count] = result;
                        inner_count += 1;
                     }//end sample level

                     if (verbose == 1) {
                        std::cout << "End Step B" << std::endl;
                     }

                     //int discard_flag = 0;
                     #pragma omp critical
                     {
                        if (1 == 1) {
                           //if (discard_flag == 0) {
                           sample_counter += 1.0;
                           int hantei     = 1;
                           int change_doc = -1;

                           for (auto itr2 = new_place_path_level.begin(); itr2 != new_place_path_level.end(); ++itr2) {
                              std::vector <int> result = itr2->second;
                              int change_place         = result[0];
                              change_doc = doc_word_topic(change_place, 0);
                              int new_path  = result[1];
                              int new_level = result[2];
                              int new_topic = result[3];
                              int old_path  = doc_word_topic(change_place, 2);
                              int old_level = doc_word_topic(change_place, 3);
                              int old_topic = path_level2topic(old_path, old_level);

                              // UPDATE
                              doc_word_topic(change_place, 2) = new_path;
                              doc_word_topic(change_place, 3) = new_level;
                              doc_word_topic(change_place, 4) = new_topic;

                              doc2level_vec[change_doc](old_level, 0) = doc2level_vec[change_doc](old_level, 0) - 1.0;
                              doc2level_vec[change_doc](new_level, 0) = doc2level_vec[change_doc](new_level, 0) + 1.0;

                              if (verbose == 1) {
                                 std::cout << "After update doc2level_vec" << std::endl;
                              }

                              if (1 == 1) {
                                 if (hantei == 1) {    // we only need to do this once
                                    hantei = 0;
                                    // SUBTRACT
                                    path2num[old_path] = path2num[old_path] - 1.0;
                                    // ADD
                                    path2num[new_path]   = path2num[new_path] + 1.0;
                                    doc2path[change_doc] = new_path;
                                    //update_num_path_num_topics(path_dict);
                                 }//if(hantei==1){

                                 if (verbose == 1) {
                                    std::cout << "change doc " << change_doc << " old level " << old_level << " new level " << new_level << std::endl;
                                 }

                                 int             change_word = doc_word_topic(change_place, 1);
                                 Eigen::MatrixXf change_x    = embedding_matrix.row(change_word);

                                 if (verbose == 1) {
                                    std::cout << "After change_word" << std::endl;
                                 }

                                 // UPDATE GLOBAL PARAMETERS
                                 ghlda_update_parameters_subtract(old_topic, change_x);

                                 if (verbose == 1) {
                                    std::cout << "After ghlda_update_parameters_subtract" << std::endl;
                                 }

                                 // UPDATE GLOBAL PARAMETERS
                                 ghlda_update_parameters_add(new_topic, change_x);

                                 if (verbose == 1) {
                                    std::cout << "After ghlda_update_parameters_add" << std::endl;
                                 }
                              }//if(path2num.find(old_path)!=path2num.end(){}else{}
                           }    //for(auto itr2=new_place_path_level.begin()
                        }       //if(discard_flag == 0){
                     }          //#pragma omp critical
                  }             //if(stop_hantei == 1)
               }                //if(path2num_thread[path_thread]>0)
            }                   //if(stop_hantei == 1)
         }                      //for(int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner)
      }                         //for(int itr = 0; itr < num_iteration; ++itr){

      if (verbose == 1) {
         std::cout << "Before Free memory" << std::endl;
      }
      // free memory
      for (int i = 0; i < num_threads; i++) {
         gsl_rng_free(r[i]);
      }
      if (verbose == 1) {
         std::cout << "After Free memory" << std::endl;
      }
   }//ghlda_collapsed_gibbs_sample_parallel

   float ghlda_posterior_predictive_third_nume(const int& topic,
                                               const float& nu, const float& kappa,
                                               std::unordered_map <int, float>& topic2num_temp,
                                               std::unordered_map <int, Eigen::MatrixXf>& topic2mu_temp,
                                               std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_temp,
                                               const int& verbose) {
      float n = 0;

      n = float(topic2num_temp[topic]);
      float       third_nume_inside = 0.0;
      float       third_nume        = 0.0;
      std::string method            = "llt";

      //if(topic2third_nume_inside.find(topic)==topic2third_nume_inside.end()){
      Eigen::MatrixXf mu           = topic2mu_temp[topic];
      Eigen::MatrixXf Sigma        = topic2Sigma_temp[topic];
      Eigen::MatrixXf Sigma_stable = Psi + Sigma + ((kappa * n) / (kappa + n)) * (mu - embedding_center).transpose() * (mu - embedding_center);
      Sigma_stable = rescale * Sigma_stable;
      if (method == "llt") {
         Eigen::LLT <Eigen::MatrixXf> llt;
         llt.compute(Sigma_stable);
         auto& U = llt.matrixL();
         for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
            if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
               third_nume_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
            }else{
               third_nume_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
            }
         }//if(method == "llt"){
         if (verbose == 1) {
            std::cout << "third_nume_inside " << third_nume_inside << std::endl;
         }
      }else{
      }
      third_nume = ((nu + n) / 2.0) * third_nume_inside;
      return(third_nume);
   }

   float ghlda_posterior_predictive_robust_fast(const int& doc,
                                                const std::vector <int>& topic_vec,
                                                const int& level,
                                                const float& nu, const float& kappa,
                                                std::unordered_map <int, float>& topic2num_temp,
                                                std::unordered_map <int, Eigen::MatrixXf>& topic2mu_temp,
                                                std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_temp,
                                                std::unordered_map <int, float>& topic2third_nume,
                                                const int& log_true, const int& verbose) {
      std::vector <int> place_vec = doc2place_vec[doc];
      std::vector <int> place_level_vec;
      for (int i = 0; i < int(place_vec.size()); ++i) {
         if (doc_word_topic(place_vec[i], 3) == level) {
            place_level_vec.push_back(place_vec[i]);
         }
      }

      const int topic = topic_vec[level];
      // KOKOGA MONDAI
      float n = 0;
      float m = 0;
      n = float(topic2num_temp[topic]);   // - float(place_vec.size());
      m = float(place_level_vec.size());

      // Actually we have to re calculate mu and sigma
      float log_prob          = 0.0;
      float prob              = 0.0;
      float first             = 0.0;
      float second_nume_denom = 0.0;
      //float third_nume_inside  = 0.0;
      float third_denom_inside = 0.0;
      float third_nume         = 0.0;
      float third_denom        = 0.0;
      float fourth_nume        = 0.0;
      float fourth_denom       = 0.0;
      // Choleskey fast
      std::string method = "llt";

      first = 0 - std::log(float(place_vec.size()) * embedding_dimension / 2.0 * std::log(M_PI));
      if (verbose == 1) {
         std::cout << "  first " << first << std::endl;
      }

      // Fast part
      //float n_m = pair_func(n,m);
      //second_nume_denom = n_m2second_nume_denom[n_m];
      float second_nume  = 0.0;
      float second_denom = 0.0;
      for (int d = 0; d < embedding_dimension; ++d) {
         second_nume  = second_nume + std::lgamma((nu + n + m - float(d)) / 2.0);
         second_denom = second_denom + std::lgamma((nu + n - float(d)) / 2.0);
      }
      second_nume_denom = second_nume - second_denom;

      // third_denom
      float           N_temp     = topic2num_temp[topic];
      Eigen::MatrixXf mu_temp    = topic2mu_temp[topic];
      Eigen::MatrixXf Sigma_temp = topic2Sigma_temp[topic];

      //third_nume  = ((nu + n) / 2.0) * third_nume_inside;
      third_nume = topic2third_nume[topic];

      int hantei = 0;
      if (N_temp == 0) {
         hantei = 1;
      }

      for (int i = 0; i < int(place_level_vec.size()); ++i) {
         int             alter_word = doc_word_topic(place_level_vec[i], 1);
         Eigen::MatrixXf alter_x    = embedding_matrix.row(alter_word);
         if (hantei == 0) {
            ghlda_update_mu_sigma(topic, alter_x, N_temp, mu_temp, Sigma_temp);
         }else{
            N_temp += 1;
            mu_temp = mu_temp + alter_x;
         }
      }

      if (hantei == 1 && N_temp > 0) {
         mu_temp = mu_temp / N_temp;
      }

      float value  = (kappa * n + m);
      float value2 = (kappa + n + m);
      float value3 = ((kappa * n + m) / (kappa + n + m));

      Eigen::MatrixXf Sigma_stable = Psi + Sigma_temp + ((kappa * n + m) / (kappa + n + m)) * (mu_temp - embedding_center).transpose() * (mu_temp - embedding_center);
      Sigma_stable = rescale * Sigma_stable;
      if (method == "llt") {
         Eigen::LLT <Eigen::MatrixXf> llt;
         llt.compute(Sigma_stable);
         auto& U = llt.matrixL();
         for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
            if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
               third_denom_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
            }else{
               third_denom_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
            }
         }
         third_denom = ((nu + m + n) / 2.0) * third_denom_inside;
      }//if(method == "llt"){

      if (verbose == 1) {
         std::cout << "  value " << value << std::endl;
         std::cout << "  value2 " << value2 << std::endl;
         std::cout << "  value3 " << value3 << std::endl;
         std::cout << "  third_nume " << third_nume << std::endl;
         std::cout << "  third_denom " << third_denom << std::endl;
      }

      fourth_nume  = (embedding_dimension / 2.0) * std::log(kappa + n);
      fourth_denom = (embedding_dimension / 2.0) * std::log(kappa + n + m);

      if (verbose == 1) {
         std::cout << "  n " << n << std::endl;
         std::cout << "  m " << m << std::endl;
         std::cout << "  fourth_nume " << fourth_nume << std::endl;
         std::cout << "  fourth_denom " << fourth_denom << std::endl;
      }

      log_prob = first + second_nume_denom + third_nume - third_denom + fourth_nume - fourth_denom;
      log_prob = log_prob - inner_rescale;

      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      return(prob);
   }

   void ghlda_collapsed_gibbs_sample_parallel_fast(const int num_iteration, const int parallel_loop, int num_threads, const int verbose) {
      if (verbose == 1) {
         std::cout << "Enter ghlda_collapsed_gibbs_sample_parallel" << std::endl;
      }

      stop_hantei = 0;
      if (num_threads < 1) {
         num_threads = omp_get_max_threads();
         std::cout << "Using " << num_threads << " threads" << std::endl;
      }

      omp_set_num_threads(num_threads);

      const gsl_rng_type *T;
      gsl_rng **          r;
      gsl_rng_env_setup();
      T = gsl_rng_mt19937;
      r = (gsl_rng **)malloc(num_threads * sizeof(gsl_rng *));
      std::random_device rand_device;
      std::mt19937       mt(rand_device());
      for (int i = 0; i < num_threads; i++) {
         r[i] = gsl_rng_alloc(T);
         gsl_rng_set(r[i], mt());
      }
      if (verbose == 1) {
         std::cout << "After gsl_rng_set" << std::endl;
      }

      for (int itr_outer = 0; itr_outer < num_iteration; ++itr_outer) {
         if (stop_hantei == 1) {
            std::cout << "ERROR: COULD NOT SAMPLE TOPIC" << std::endl;
            break;
         }

         #pragma omp parallel for
         for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
            if (stop_hantei != 1) {
               int thread_id   = omp_get_thread_num();
               int doc_thread  = int(floor(float(num_docs) * gsl_rng_uniform(r[thread_id])));
               int path_thread = doc2path[doc_thread];
               if (verbose == 1) {
                  std::cout << "After path_thread" << std::endl;
               }

               //Used in path sampling
               std::unordered_map <int, float>           path2num_thread    = path2num;
               std::unordered_map <int, float>           topic2num_thread   = topic2num;
               std::unordered_map <int, Eigen::MatrixXf> topic2mu_thread    = topic2mu;
               std::unordered_map <int, Eigen::MatrixXf> topic2Sigma_thread = topic2Sigma;
               //Used in level sampling
               std::unordered_map <int, float>           topic2num_thread_2   = topic2num;
               std::unordered_map <int, Eigen::MatrixXf> topic2mu_thread_2    = topic2mu;
               std::unordered_map <int, Eigen::MatrixXf> topic2Sigma_thread_2 = topic2Sigma;

               if (1 == 1) {
                  // Subtract Path
                  path2num_thread[path_thread] = path2num_thread[path_thread] - 1.0;

                  // Minus document assignment
                  std::vector <int> place_vec_thread = doc2place_vec[doc_thread];
                  int true_path = -1;

                  for (int i = 0; i < int(place_vec_thread.size()); ++i) {
                     int             alter_word  = doc_word_topic(place_vec_thread[i], 1);
                     Eigen::MatrixXf alter_x     = embedding_matrix.row(alter_word);
                     int             alter_topic = doc_word_topic(place_vec_thread[i], 4);
                     ghlda_update_parameters_subtract_thread(alter_topic, alter_x, topic2num_thread, topic2mu_thread, topic2Sigma_thread);
                  }// for(int i=0;i<int(place_vec_thread.size());++i){

                  if (verbose == 1) {
                     std::cout << "After ghlda_update_parameters_subtract_thread" << std::endl;
                  }

                  std::map <int, std::vector <int> > new_path_dict = path_dict;

                  if (verbose == 1) {
                     for (auto itr = new_path_dict.begin(); itr != new_path_dict.end(); ++itr) {
                        std::vector <int> v1          = itr->second;
                        std::string       temp_string = std::to_string(itr->first) + ":";
                        for (int i = 0; i < int(v1.size()); ++i) {
                           temp_string += std::to_string(v1[i]) + ",";
                        }
                        std::cout << temp_string << std::endl;
                     }
                  }//if(verbose == 1){

                  // Step A: sample path for a document
                  std::string               prob_string = "";
                  float                     agg_topic   = 0;
                  float                     max_val     = -9999999999999;
                  std::vector <int>         path_id_vec;
                  std::vector <float>       ratio;
                  std::vector <std::string> topic_string_vec;

                  // Calculate First
                  std::unordered_map <int, float> topic2third_nume;
                  for (auto itr = topic2num.begin(); itr != topic2num.end(); ++itr) {
                     if (itr->first > 0) {
                        int topic_thread = itr->first;
                        topic2third_nume[topic_thread] = ghlda_posterior_predictive_third_nume(topic_thread, nu, kappa,
                                                                                               topic2num_thread, topic2mu_thread, topic2Sigma_thread, 0);
                     }
                  }

                  for (auto itr = new_path_dict.begin(); itr != new_path_dict.end(); ++itr) {
                     int path_id = itr->first;
                     path_id_vec.push_back(path_id);
                     std::vector <int> topic_vec = itr->second;
                     float             temp      = 0.0;
                     float             temp_in   = 0.0;
                     prob_string = "";
                     for (int j = 0; j < int(topic_vec.size()); ++j) {
                        temp_in = ghlda_posterior_predictive_robust_fast(doc_thread,
                                                                         topic_vec, j, nu, kappa, topic2num_thread, topic2mu_thread, topic2Sigma_thread,
                                                                         topic2third_nume, 1, 0);
                        temp += temp_in;
                        float temp_val = topic2num_thread[topic_vec[j]];
                        prob_string = prob_string + "," + std::to_string(topic_vec[j]) + "-" + std::to_string(temp_val);
                     }
                     topic_string_vec.push_back(prob_string);
                     if (path2num_thread.find(path_id) != path2num_thread.end()) {
                        temp += std::log(path2num_thread[path_id] + path_gamma);
                     }else{
                        temp += std::log(path_gamma);
                     }
                     ratio.push_back(temp);
                     if (max_val < temp) {
                        max_val = temp;
                     }
                  }

                  if (verbose == 1) {
                     std::cout << "After Step A before sample" << std::endl;
                  }

                  // sample
                  prob_string = "";
                  for (int i = 0; i < int(ratio.size()); ++i) {
                     ratio[i]    = ratio[i] - max_val;
                     prob_string = prob_string + ":" + std::to_string(i) + "-" + topic_string_vec[i] + "," + std::to_string(ratio[i]) + "\n";
                     ratio[i]    = std::exp(ratio[i]);
                     agg_topic  += ratio[i];
                  }
                  double prob[int(ratio.size())]; // Probability array
                  for (int i = 0; i < int(ratio.size()); ++i) {
                     prob[i] = double(ratio[i] / agg_topic);
                  }
                  unsigned int mult_op[ratio.size()];
                  int          num_sample = 1;
                  gsl_ran_multinomial(r[thread_id], int(ratio.size()), num_sample, prob, mult_op);
                  int new_path = -1;
                  for (int i = 0; i < int(ratio.size()); ++i) {
                     if (mult_op[i] == 1) {
                        new_path = path_id_vec[i];
                        break;
                     }
                  }

                  if (verbose == 1) {
                     std::cout << "doc num " << doc_thread << std::endl;
                     std::cout << "True " << true_path << " Sample " << new_path << std::endl;
                     std::cout << prob_string << std::endl;
                  }
                  if (new_path < 0) {
                     #pragma omp critical
                     {
                        stop_hantei = 1;
                        std::cout << "ERROR STOP PATH" << std::endl;
                     }
                  }// if(new_path < 0){

                  // new_path
                  std::vector <int> new_path_vec = new_path_dict[new_path];

                  // Step B: sample level
                  if (stop_hantei != 1) {
                     // COPY
                     std::unordered_map <long long, Eigen::MatrixXf> doc2level_vec_thread = doc2level_vec;

                     // Create place_vec_thread shuffled
                     std::map <int, std::vector <int> > new_place_path_level;
                     std::random_device seed_gen;
                     std::mt19937       engine(seed_gen());
                     std::shuffle(place_vec_thread.begin(), place_vec_thread.end(), engine);
                     int inner_count = 0;

                     // for each word position
                     for (int itr_place = 0; itr_place < int(place_vec_thread.size()); ++itr_place) {
                        int             place     = place_vec_thread[itr_place];
                        int             word      = doc_word_topic(place, 1);
                        Eigen::MatrixXf x         = embedding_matrix.row(word);
                        int             old_level = doc_word_topic(place, 3);
                        int             old_topic = doc_word_topic(place, 4);

                        //SUBTRACT
                        ghlda_update_parameters_subtract_thread(old_topic, x,
                                                                topic2num_thread_2, topic2mu_thread_2, topic2Sigma_thread_2);

                        int hantei_update_doc2level = 0;
                        if (doc2level_vec_thread[doc_thread](old_level, 0) > 0) {
                           doc2level_vec_thread[doc_thread](old_level, 0) = doc2level_vec_thread[doc_thread](old_level, 0) - 1.0;
                           hantei_update_doc2level = 1;
                        }

                        // Create level topic ratio
                        std::vector <float> level_ratio;
                        float agg_level = 0.0;
                        if (level_allocation_type == 0) {// Pure LDA
                           for (int i = 0; i < int(new_path_vec.size()); ++i) {
                              float temp = alpha_level_vec(i) + doc2level_vec_thread[doc_thread](i, 0);
                              level_ratio.push_back(temp);
                              agg_level += temp;
                           }
                        }else{// HLDA
                           for (int i = 0; i < int(new_path_vec.size()); ++i) {
                              float first       = 1.0;
                              float first_nume  = hyper_m * hyper_pi + doc2level_vec_thread[doc_thread](i, 0);
                              float first_denom = hyper_pi;
                              for (int j = i; j < int(new_path_vec.size()); ++j) {
                                 first_denom += doc2level_vec_thread[doc_thread](j, 0);
                              }
                              first = first_nume / first_denom;
                              float second = 1.0;
                              for (int j = 0; j < i; ++j) {
                                 float second_nume_inside = 0.0;
                                 for (int k = j + 1; k < int(new_path_vec.size()); ++k) {
                                    second_nume_inside += doc2level_vec_thread[doc_thread](k, 0);
                                 }
                                 float second_nume         = (1.0 - hyper_m) * hyper_pi + second_nume_inside;
                                 float second_denom_inside = 0.0;
                                 for (int k = j; k < int(new_path_vec.size()); ++k) {
                                    second_denom_inside += doc2level_vec_thread[doc_thread](k, 0);
                                 }
                                 float second_denom = hyper_pi + second_denom_inside;
                                 second = second * (second_nume / second_denom);
                              }
                              float temp = first * second;
                              level_ratio.push_back(temp);
                              agg_level += temp;
                           }
                        }//if(level_allocation_type == 0){}else{}

                        float agg_topic = 0;
                        max_val = -9999999999999;
                        std::vector <float> ratio;
                        for (int j = 0; j < int(new_path_vec.size()); ++j) {
                           int   topic      = new_path_vec[j];
                           float log_prob_a = std::log(level_ratio[j]) - std::log(agg_level);
                           float log_prob_b = ghlda_posterior_predictive_single_robust(place, j, topic, nu, kappa,
                                                                                       topic2num_thread_2, topic2mu_thread_2,
                                                                                       topic2Sigma_thread_2, 1, verbose);
                           float log_prob = log_prob_a + log_prob_b;
                           ratio.push_back(log_prob);
                           if (max_val < ratio[j]) {
                              max_val = ratio[j];
                           }
                        }
                        agg_topic = 0;
                        for (int j = 0; j < int(ratio.size()); ++j) {
                           ratio[j]   = ratio[j] - max_val;
                           ratio[j]   = std::exp(ratio[j]);
                           agg_topic += ratio[j];
                        }
                        // sample
                        double prob[ratio.size()]; // Probability array
                        for (int j = 0; j < int(ratio.size()); ++j) {
                           prob[j] = double(ratio[j] / agg_topic);
                        }
                        unsigned int mult_op[int(ratio.size())];
                        int          num_sample = 1;
                        gsl_ran_multinomial(r[thread_id], int(ratio.size()), num_sample, prob, mult_op);
                        int new_level = -1;
                        int new_topic = -1;
                        for (int j = 0; j < int(ratio.size()); ++j) {
                           if (mult_op[j] == 1) {
                              new_level = j;
                              new_topic = new_path_vec[new_level];
                              break;
                           }
                        }

                        // ADD
                        ghlda_update_parameters_add_thread(new_topic, x, topic2num_thread_2, topic2mu_thread_2, topic2Sigma_thread_2);

                        if (hantei_update_doc2level == 1) {
                           doc2level_vec_thread[doc_thread](new_level, 0) = doc2level_vec_thread[doc_thread](new_level, 0) + 1.0;
                        }

                        std::vector <int> result;
                        result.push_back(place);
                        result.push_back(new_path);
                        result.push_back(new_level);
                        result.push_back(new_topic);
                        new_place_path_level[inner_count] = result;
                        inner_count += 1;
                     }//end sample level

                     if (verbose == 1) {
                        std::cout << "End Step B" << std::endl;
                     }

                     //int discard_flag = 0;
                     #pragma omp critical
                     {
                        if (1 == 1) {
                           //if (discard_flag == 0) {
                           sample_counter += 1.0;
                           int hantei     = 1;
                           int change_doc = -1;

                           for (auto itr2 = new_place_path_level.begin(); itr2 != new_place_path_level.end(); ++itr2) {
                              std::vector <int> result = itr2->second;
                              int change_place         = result[0];
                              change_doc = doc_word_topic(change_place, 0);
                              int new_path  = result[1];
                              int new_level = result[2];
                              int new_topic = result[3];
                              int old_path  = doc_word_topic(change_place, 2);
                              int old_level = doc_word_topic(change_place, 3);
                              int old_topic = path_level2topic(old_path, old_level);

                              // UPDATE
                              doc_word_topic(change_place, 2) = new_path;
                              doc_word_topic(change_place, 3) = new_level;
                              doc_word_topic(change_place, 4) = new_topic;

                              doc2level_vec[change_doc](old_level, 0) = doc2level_vec[change_doc](old_level, 0) - 1.0;
                              doc2level_vec[change_doc](new_level, 0) = doc2level_vec[change_doc](new_level, 0) + 1.0;

                              if (verbose == 1) {
                                 std::cout << "After update doc2level_vec" << std::endl;
                              }

                              if (1 == 1) {
                                 if (hantei == 1) {    // we only need to do this once
                                    hantei = 0;
                                    // SUBTRACT
                                    path2num[old_path] = path2num[old_path] - 1.0;
                                    // ADD
                                    path2num[new_path]   = path2num[new_path] + 1.0;
                                    doc2path[change_doc] = new_path;
                                    //update_num_path_num_topics(path_dict);
                                 }//if(hantei==1){

                                 if (verbose == 1) {
                                    std::cout << "change doc " << change_doc << " old level " << old_level << " new level " << new_level << std::endl;
                                 }

                                 int             change_word = doc_word_topic(change_place, 1);
                                 Eigen::MatrixXf change_x    = embedding_matrix.row(change_word);

                                 if (verbose == 1) {
                                    std::cout << "After change_word" << std::endl;
                                 }

                                 // UPDATE GLOBAL PARAMETERS
                                 ghlda_update_parameters_subtract(old_topic, change_x);

                                 if (verbose == 1) {
                                    std::cout << "After ghlda_update_parameters_subtract" << std::endl;
                                 }

                                 // UPDATE GLOBAL PARAMETERS
                                 ghlda_update_parameters_add(new_topic, change_x);

                                 if (verbose == 1) {
                                    std::cout << "After ghlda_update_parameters_add" << std::endl;
                                 }
                              } //if(path2num.find(old_path)!=path2num.end(){}else{}
                           }    //for(auto itr2=new_place_path_level.begin()
                        }       //if(discard_flag == 0){
                     }          //#pragma omp critical
                  }             //if(stop_hantei == 1)
               }                //if(path2num_thread[path_thread]>0)
            }                   //if(stop_hantei == 1)
         }                      //for(int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner)
      }                         //for(int itr = 0; itr < num_iteration; ++itr){

      if (verbose == 1) {
         std::cout << "Before Free memory" << std::endl;
      }
      // free memory
      for (int i = 0; i < num_threads; i++) {
         gsl_rng_free(r[i]);
      }
      if (verbose == 1) {
         std::cout << "After Free memory" << std::endl;
      }
   }//ghlda_collapsed_gibbs_sample_parallel_fast

   // GHLDA-Fixed END //

   // GHLDA-ncrp START //
   std::map <int, std::vector <int> > new_path_dict_debug, extended_path_dict_debug;

   void ghlda_ncrp_calc_tables_parameters_from_assign() {
      if (embedding_dimension == 0) {
         std::cout << "Please set embedding_dimension" << std::endl;
      }else{
         doc2level_vec.clear();
         word2count.clear();
         word2topic_vec.clear();
         for (auto itr = topic2num.begin(); itr != topic2num.end(); ++itr) {
            topic2num[itr->first] = 0.0;
         }
         for (auto itr = topic2mu.begin(); itr != topic2mu.end(); ++itr) {
            topic2mu[itr->first] = zero_mu;
         }
         for (auto itr = topic2Sigma.begin(); itr != topic2Sigma.end(); ++itr) {
            topic2Sigma[itr->first] = zero_Sigma;
         }

         topic2num[-1]   = 0.0;
         topic2mu[-1]    = zero_mu;
         topic2Sigma[-1] = zero_Sigma;

         for (int i = 0; i < doc_word_topic.rows(); ++i) {
            int doc   = doc_word_topic(i, 0);
            int word  = doc_word_topic(i, 1);
            int level = doc_word_topic(i, 3);
            int topic = doc_word_topic(i, 4);

            Eigen::MatrixXf x = embedding_matrix.row(word);
            // Create tables
            if (doc2level_vec.find(doc) == doc2level_vec.end()) {
               doc2level_vec[doc] = level_gamma * Eigen::MatrixXf::Ones(num_depth + 1000, 1);
            }
            doc2level_vec[doc](level, 0) += 1.0;
            if (word2count.find(word) == word2count.end()) {
               word2count[word] = 1.0;
            }else{
               word2count[word] += 1.0;
            }
            if (word2topic_vec.find(word) == word2topic_vec.end()) {
               word2topic_vec[word] = Eigen::MatrixXf::Zero(num_topics + 1000, 1);
            }
            num_not_assigned = 0.0;
            if (topic != -1) {
               word2topic_vec[word](topic, 0) += 1.0;
            }else{
               num_not_assigned += 1.0;
            }

            if (topic != -1) {
               // Create Parameters
               if (topic2num.find(topic) == topic2num.end()) {
                  topic2num[topic] = 1.0;
                  topic2mu[topic]  = x;
               }else{
                  topic2num[topic] += 1.0;
                  topic2mu[topic]  += x;
               }
            }
         }

         // Normalize
         for (auto itr = topic2num.begin(); itr != topic2num.end(); ++itr) {
            if (itr->second > 0) {
               topic2mu[itr->first] = topic2mu[itr->first] / itr->second;
            }
         }

         // Variance
         for (int i = 0; i < doc_word_topic.rows(); ++i) {
            int             word  = doc_word_topic(i, 1);
            int             topic = doc_word_topic(i, 4);
            Eigen::MatrixXf x     = embedding_matrix.row(word);
            if (topic != -1) {
               if (topic2Sigma.find(topic) == topic2Sigma.end()) {
                  Eigen::MatrixXf outer_matrix = (x - topic2mu[topic]).transpose() * (x - topic2mu[topic]);
                  topic2Sigma[topic] = outer_matrix;
               }else{
                  Eigen::MatrixXf outer_matrix = (x - topic2mu[topic]).transpose() * (x - topic2mu[topic]);
                  topic2Sigma[topic] += outer_matrix;
               }
            }
         }

         // Erase unnecessary THIS IS THE ONLY DIFFERENCE
         std::map <int, int> used_topic_id_t;
         for (auto itr = path_dict.begin(); itr != path_dict.end(); ++itr) {
            std::vector <int> path_vec_t  = itr->second;
            std::string       temp_string = "";
            for (int j = 0; j < int(path_vec_t.size()); ++j) {
               if (used_topic_id_t.find(path_vec_t[j]) == used_topic_id_t.end()) {
                  used_topic_id_t[path_vec_t[j]] = 1;
               }
            }
         }
         // SAFER VERSION
         std::unordered_map <int, float>           topic2num_t;
         std::unordered_map <int, Eigen::MatrixXf> topic2mu_t, topic2Sigma_t;
         topic2num_t[-1]   = 0.0;
         topic2mu_t[-1]    = zero_mu;
         topic2Sigma_t[-1] = zero_Sigma;
         for (auto itr = topic2num.begin(); itr != topic2num.end(); ++itr) {
            int topic = itr->first;
            if (itr->second > 0 && used_topic_id_t.find(topic) != used_topic_id_t.end()) {
               topic2num_t[topic]   = itr->second;
               topic2mu_t[topic]    = topic2mu[topic];
               topic2Sigma_t[topic] = topic2Sigma_t[topic];
            }
         }
         topic2num.clear();
         topic2num = topic2num_t;
         topic2mu.clear();
         topic2mu = topic2mu_t;
         topic2Sigma.clear();
         topic2Sigma = topic2Sigma_t;
         // OLD VERSION
         //for (auto itr = topic2num.begin(); itr != topic2num.end(); ++itr) {
         //    int topic = itr->first;
         //    if (topic != -1 && used_topic_id_t.find(topic) == used_topic_id_t.end()) {
         //       topic2num.erase(topic);
         //       topic2mu.erase(topic);
         //       topic2Sigma.erase(topic);
         //    }
         //}
      }
   }

   void update_num_path_num_topics(std::map <int, std::vector <int> >& path_dict_t) {
      std::unordered_map <int, int> used_path_id_t, used_topic_id_t;
      num_path   = 0;
      num_topics = 0;
      for (auto itr = path_dict_t.begin(); itr != path_dict_t.end(); ++itr) {
         if (used_path_id_t.find(itr->first) == used_path_id_t.end()) {
            used_path_id_t[itr->first] = 1;
            num_path += 1;
         }
         std::vector <int> path_vec_t = itr->second;
         for (int j = 0; j < int(path_vec_t.size()); ++j) {
            if (used_topic_id_t.find(path_vec_t[j]) == used_topic_id_t.end()) {
               used_topic_id_t[path_vec_t[j]] = 1;
               num_topics += 1;
            }
         }
      }
   }

   void update_num_path_num_topics_etc(const std::map <int, std::vector <int> >& path_dict_t, std::map <int, int>& used_path_id_t, std::map <int, int>& used_topic_id_t, std::map <std::string, int>& path_temp_t,
                                       int& num_path_t, int& num_topics_t, int& num_new_path_t) {
      num_path_t     = 0;
      num_topics_t   = 0;
      num_new_path_t = 0;
      for (auto itr = path_dict_t.begin(); itr != path_dict_t.end(); ++itr) {
         if (used_path_id_t.find(itr->first) == used_path_id_t.end()) {
            used_path_id_t[itr->first] = 1;
            num_path_t += 1;
         }
         std::vector <int> path_vec_t  = itr->second;
         std::string       temp_string = "";
         for (int j = 0; j < int(path_vec_t.size()); ++j) {
            if (used_topic_id_t.find(path_vec_t[j]) == used_topic_id_t.end()) {
               used_topic_id_t[path_vec_t[j]] = 1;
               num_topics_t += 1;
            }
            temp_string += std::to_string(path_vec_t[j]) + ",";
            std::string temp_string_2 = temp_string + "new";
            if (int(path_vec_t.size()) == max_depth_allowed && (j == int(path_vec_t.size()) - 1)) {
               //pass
            }else{
               if (path_temp_t.find(temp_string_2) == path_temp_t.end()) {
                  path_temp_t[temp_string_2] = 1;
                  num_new_path_t            += 1;
               }
            }
         }
      }
   }

   float ghlda_ncrp_posterior_predictive_robust(const int& doc,
                                                const std::vector <int>& topic_vec,
                                                const int& level,
                                                const int& last_topic,
                                                const float& nu, const float& kappa,
                                                std::unordered_map <int, float>& topic2num_temp,
                                                std::unordered_map <int, Eigen::MatrixXf>& topic2mu_temp,
                                                std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_temp,
                                                std::unordered_map <int, float> topic2third_nume_inside,
                                                const int& log_true, const int& verbose) {
      std::vector <int> place_vec = doc2place_vec[doc];

      std::vector <int> place_level_vec;
      for (int i = 0; i < int(place_vec.size()); ++i) {
         if (last_topic != 1) {
            if (doc_word_topic(place_vec[i], 3) == level) {
               place_level_vec.push_back(place_vec[i]);
            }
         }else{
            if (doc_word_topic(place_vec[i], 3) >= level) {
               place_level_vec.push_back(place_vec[i]);
            }
         }
      }

      const int topic = topic_vec[level];
      float     n     = 0;
      float     m     = 0;
      n = float(topic2num_temp[topic]);
      m = float(place_level_vec.size());

      // Actually we have to re calculate mu and sigma
      float log_prob           = 0.0;
      float prob               = 0.0;
      float first              = 0.0;
      float second_nume        = 0.0;
      float second_denom       = 0.0;
      float third_nume_inside  = 0.0;
      float third_denom_inside = 0.0;
      float third_nume         = 0.0;
      float third_denom        = 0.0;
      float fourth_nume        = 0.0;
      float fourth_denom       = 0.0;
      // Choleskey fast
      std::string method = "llt";

      first = 0 - std::log(float(place_vec.size()) * embedding_dimension / 2.0 * std::log(M_PI));
      if (verbose == 1) {
         std::cout << "  first " << first << std::endl;
      }

      for (int d = 0; d < embedding_dimension; ++d) {
         second_nume  = second_nume + std::lgamma((nu + n + m - float(d)) / 2.0);
         second_denom = second_denom + std::lgamma((nu + n - float(d)) / 2.0);
      }

      if (verbose == 1) {
         std::cout << "  second_nume " << second_nume << " " << std::to_string(nu) << std::endl;
         std::cout << "  second_denom " << second_denom << " " << std::to_string(nu) << std::endl;
      }

      std::unordered_map <int, float> thres2third_nume_inside;

      if (1 == 1) {
         if (topic2mu_temp.find(topic) == topic2mu_temp.end()) {
            std::cout << "Mu topic " << topic << " does not exist" << std::endl;
         }

         if (topic2Sigma_temp.find(topic) == topic2Sigma_temp.end()) {
            std::cout << "Sigma topic " << topic << " does not exist" << std::endl;
         }

         Eigen::MatrixXf mu           = topic2mu_temp[topic];
         Eigen::MatrixXf Sigma        = topic2Sigma_temp[topic];
         Eigen::MatrixXf Sigma_stable = Psi + Sigma + ((kappa * n) / (kappa + n)) * (mu - embedding_center).transpose() * (mu - embedding_center);
         Sigma_stable = rescale * Sigma_stable;

         if (method == "llt") {
            Eigen::LLT <Eigen::MatrixXf> llt;
            llt.compute(Sigma_stable);
            auto& U = llt.matrixL();
            for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
               if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
                  third_nume_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
               }else{
                  third_nume_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
               }
               thres2third_nume_inside[i + 1] = third_nume_inside;
            }
         }        //if(method == "llt"){
         if (verbose == 1) {
            std::cout << "third_nume_inside " << third_nume_inside << std::endl;
         }
         topic2third_nume_inside[topic] = third_nume_inside;
      }else{
         third_nume_inside = topic2third_nume_inside[topic];
      }

      // third_denom
      float           N_temp     = topic2num_temp[topic];
      Eigen::MatrixXf mu_temp    = topic2mu_temp[topic];
      Eigen::MatrixXf Sigma_temp = topic2Sigma_temp[topic];

      int hantei = 0;
      if (N_temp == 0) {
         hantei = 1;
      }

      for (int i = 0; i < int(place_level_vec.size()); ++i) {
         int             alter_word = doc_word_topic(place_level_vec[i], 1);
         Eigen::MatrixXf alter_x    = embedding_matrix.row(alter_word);
         if (hantei == 0) {
            ghlda_update_mu_sigma(topic, alter_x, N_temp, mu_temp, Sigma_temp);
         }else{
            N_temp += 1;
            mu_temp = mu_temp + alter_x;
         }
      }

      if (hantei == 1 && N_temp > 0) {
         mu_temp = mu_temp / N_temp;
      }

      float value  = (kappa * n + m);
      float value2 = (kappa + n + m);
      float value3 = ((kappa * n + m) / (kappa + n + m));

      Eigen::MatrixXf Sigma_stable = Psi + Sigma_temp + ((kappa * n + m) / (kappa + n + m)) * (mu_temp - embedding_center).transpose() * (mu_temp - embedding_center);
      Sigma_stable = rescale * Sigma_stable;
      if (method == "llt") {
         Eigen::LLT <Eigen::MatrixXf> llt;
         llt.compute(Sigma_stable);
         auto& U = llt.matrixL();
         for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
            if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
               third_denom_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
            }else{
               third_denom_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
            }
         }
         third_nume  = ((nu + n) / 2.0) * third_nume_inside;
         third_denom = ((nu + m + n) / 2.0) * third_denom_inside;
      }   //if(method == "llt"){

      if (verbose == 1) {
         std::cout << "  value " << value << std::endl;
         std::cout << "  value2 " << value2 << std::endl;
         std::cout << "  value3 " << value3 << std::endl;
         std::cout << "  third_nume " << third_nume << std::endl;
         std::cout << "  third_denom " << third_denom << std::endl;
      }

      fourth_nume  = (embedding_dimension / 2.0) * std::log(kappa + n);
      fourth_denom = (embedding_dimension / 2.0) * std::log(kappa + n + m);

      if (verbose == 1) {
         std::cout << "  n " << n << std::endl;
         std::cout << "  m " << m << std::endl;
         std::cout << "  fourth_nume " << fourth_nume << std::endl;
         std::cout << "  fourth_denom " << fourth_denom << std::endl;
      }

      log_prob = first + second_nume - second_denom + third_nume - third_denom + fourth_nume - fourth_denom;
      log_prob = log_prob - inner_rescale;

      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      return(prob);
   }

   void ghlda_ncrp_collapsed_gibbs_sample_parallel(const int num_iteration, const int parallel_loop, int num_threads, const int verbose) {
      if (num_threads < 1) {
         num_threads = omp_get_max_threads();
         std::cout << "Using " << num_threads << " threads" << std::endl;
      }
      omp_set_num_threads(num_threads);

      const gsl_rng_type *T;
      gsl_rng **          r;
      gsl_rng_env_setup();
      T = gsl_rng_mt19937;
      r = (gsl_rng **)malloc(num_threads * sizeof(gsl_rng *));
      std::random_device rand_device;
      std::mt19937       mt(rand_device());
      for (int i = 0; i < num_threads; i++) {
         r[i] = gsl_rng_alloc(T);
         gsl_rng_set(r[i], mt());
      }

      for (int itr_outer = 0; itr_outer < num_iteration; ++itr_outer) {
         if (stop_hantei == 1) {
            std::cout << "ERROR: COULD NOT SAMPLE TOPIC" << std::endl;
            break;
         }

         #pragma omp parallel for
         for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
            if (stop_hantei != 1) {
               int thread_id   = omp_get_thread_num();
               int doc_thread  = int(floor(float(num_docs) * gsl_rng_uniform(r[thread_id])));
               int path_thread = doc2path[doc_thread];

               //Used in path sampling
               std::unordered_map <int, float>           path2num_thread    = path2num;
               std::unordered_map <int, float>           topic2num_thread   = topic2num;
               std::unordered_map <int, Eigen::MatrixXf> topic2mu_thread    = topic2mu;
               std::unordered_map <int, Eigen::MatrixXf> topic2Sigma_thread = topic2Sigma;
               //Used in level sampling
               std::unordered_map <int, float>                 topic2num_thread_2   = topic2num;
               std::unordered_map <int, Eigen::MatrixXf>       topic2mu_thread_2    = topic2mu;
               std::unordered_map <int, Eigen::MatrixXf>       topic2Sigma_thread_2 = topic2Sigma;
               std::unordered_map <long long, Eigen::MatrixXf> doc2level_vec_thread = doc2level_vec;

               if (path2num_thread[path_thread] > 0) {
                  //if(1==1){
                  path2num_thread[path_thread] = path2num_thread[path_thread] - 1.0;
                  std::vector <int> place_vec_thread = doc2place_vec[doc_thread];

                  for (int i = 0; i < int(place_vec_thread.size()); ++i) {
                     int             alter_word  = doc_word_topic(place_vec_thread[i], 1);
                     Eigen::MatrixXf alter_x     = embedding_matrix.row(alter_word);
                     int             alter_topic = doc_word_topic(place_vec_thread[i], 4);
                     ghlda_update_parameters_subtract_thread(alter_topic, alter_x, topic2num_thread, topic2mu_thread, topic2Sigma_thread);
                  } // for(int i=0;i<int(place_vec_thread.size());++i){

                  std::map <int, std::vector <int> > new_path_dict = path_dict;

                  // BNP //
                  std::map <int, int>         used_path_id, used_topic_id;
                  std::map <std::string, int> path_temp;
                  int num_path_thread     = 0;
                  int num_topics_thread   = 0;
                  int num_new_path_thread = 0;

                  update_num_path_num_topics_etc(new_path_dict, used_path_id, used_topic_id, path_temp,
                                                 num_path_thread, num_topics_thread, num_new_path_thread);

                  std::vector <int> new_path_id, new_topic_id;
                  for (int i = 0; i < (num_path_thread + num_path_thread * max_depth_allowed * num_new_path_thread); ++i) {
                     if (used_path_id.find(i) == used_path_id.end()) {
                        new_path_id.push_back(i);
                     }
                  } //for(int i=0;i<num_path+num_new_path;
                  for (int i = 0; i < (num_topics_thread + num_path_thread * max_depth_allowed * num_new_path_thread); ++i) {
                     if (used_topic_id.find(i) == used_topic_id.end()) {
                        new_topic_id.push_back(i);
                     }
                  } //for(int i=0;i<num_topics+num_new_path;
                  int count_path  = 0;
                  int count_topic = 0;
                  for (auto itr = path_temp.begin(); itr != path_temp.end(); ++itr) {
                     std::string temp_string      = itr->first;
                     std::vector <std::string> v1 = Split(temp_string, ',');
                     std::vector <int>         v2;
                     int temp_int = -1;
                     for (int i = 0; i < int(v1.size()); ++i) {
                        if (v1[i] != "new") {
                           temp_int = std::atoi(v1[i].c_str());
                           v2.push_back(temp_int);
                        }else{
                           temp_int = new_topic_id[count_topic]; // new topic id
                           v2.push_back(temp_int);
                           count_topic += 1;
                           if (topic2num_thread.find(temp_int) == topic2num_thread.end()) {
                              topic2num_thread[temp_int]   = 0.0;
                              topic2mu_thread[temp_int]    = zero_mu;
                              topic2Sigma_thread[temp_int] = zero_Sigma;
                           }
                           if (topic2num_thread_2.find(temp_int) == topic2num_thread_2.end()) {
                              topic2num_thread_2[temp_int]   = 0.0;
                              topic2mu_thread_2[temp_int]    = zero_mu;
                              topic2Sigma_thread_2[temp_int] = zero_Sigma;
                           }
                        }
                     } //for(int i = 0;i < int(v1.size());++i){
                     temp_int = new_path_id[count_path];
                     new_path_dict[temp_int] = v2;
                     count_path += 1;
                  } //for(auto itr=path_temp.begin();

                  std::map <int, std::vector <int> > extended_path_dict;
                  for (auto itr = new_path_dict.begin(); itr != new_path_dict.end(); ++itr) {
                     std::vector <int> path_vec = itr->second;
                     extended_path_dict[itr->first] = itr->second;
                     int temp_int = -1;
                     if (int(path_vec.size()) < max_depth_allowed) {
                        int going_to_add = max_depth_allowed - int(path_vec.size());
                        for (int l = 0; l < going_to_add; ++l) {
                           temp_int = new_topic_id[count_topic];    // new topic id
                           path_vec.push_back(temp_int);
                           count_topic += 1;
                           if (topic2num_thread.find(temp_int) == topic2num_thread.end()) { // used in path sampling
                              topic2num_thread[temp_int]   = 0.0;
                              topic2mu_thread[temp_int]    = zero_mu;
                              topic2Sigma_thread[temp_int] = zero_Sigma;
                           }
                           if (topic2num_thread_2.find(temp_int) == topic2num_thread_2.end()) { // used in level sampling
                              topic2num_thread_2[temp_int]   = 0.0;
                              topic2mu_thread_2[temp_int]    = zero_mu;
                              topic2Sigma_thread_2[temp_int] = zero_Sigma;
                           }
                           temp_int = new_path_id[count_path];
                           extended_path_dict[temp_int] = path_vec;
                           count_path += 1;
                        }
                     }
                  }

                  if (thread_id == 0) {
                     new_path_dict_debug      = new_path_dict;
                     extended_path_dict_debug = extended_path_dict;
                  }

                  // Step A: sample path for a document
                  float                           agg_topic = 0;
                  float                           max_val   = -9999999999999;
                  std::vector <int>               path_id_vec;
                  std::vector <float>             ratio;
                  std::vector <std::string>       topic_string_vec;
                  std::unordered_map <int, float> topic2third_nume_inside;
                  for (auto itr = new_path_dict.begin(); itr != new_path_dict.end(); ++itr) {
                     int path_id = itr->first;
                     path_id_vec.push_back(path_id);
                     std::vector <int> topic_vec = itr->second;
                     float             temp      = 0.0;
                     float             temp_in   = 0.0;
                     for (int j = 0; j < int(topic_vec.size()); ++j) {
                        int last_hantei = 0;
                        if (j == int(topic_vec.size()) - 1) {
                           last_hantei = 1;
                        }
                        temp_in = ghlda_ncrp_posterior_predictive_robust(doc_thread,
                                                                         topic_vec, j, last_hantei, nu, kappa, topic2num_thread, topic2mu_thread,
                                                                         topic2Sigma_thread, topic2third_nume_inside, 1, 0);
                        temp += temp_in;
                     }
                     if (path2num_thread.find(path_id) != path2num_thread.end()) {
                        temp += std::log(path2num_thread[path_id] + path_gamma);
                     }else{
                        temp += std::log(path_gamma);
                     }
                     ratio.push_back(temp);
                     if (max_val < temp) {
                        max_val = temp;
                     }
                  }
                  // sample
                  for (int i = 0; i < int(ratio.size()); ++i) {
                     ratio[i]   = ratio[i] - max_val;
                     ratio[i]   = std::exp(ratio[i]);
                     agg_topic += ratio[i];
                  }
                  double prob[int(ratio.size())];  // Probability array
                  for (int i = 0; i < int(ratio.size()); ++i) {
                     prob[i] = double(ratio[i] / agg_topic);
                  }
                  unsigned int mult_op[ratio.size()];
                  int          num_sample = 1;
                  gsl_ran_multinomial(r[thread_id], int(ratio.size()), num_sample, prob, mult_op);
                  int new_path = -1;
                  for (int i = 0; i < int(ratio.size()); ++i) {
                     if (mult_op[i] == 1) {
                        new_path = path_id_vec[i];
                        break;
                     }
                  }

                  if (new_path < 0) {
                     #pragma omp critical
                     {
                        stop_hantei = 1;
                        std::cout << "ERROR STOP PATH" << std::endl;
                     }
                  } // if(new_path < 0){

                  // new_path
                  std::vector <int> new_path_vec;

                  // Step B: sample level
                  if (stop_hantei != 1) {
                     new_path_vec = new_path_dict[new_path];
                     // Create place_vec_thread shuffled
                     std::map <int, std::vector <int> > new_place_path_level;
                     std::random_device seed_gen;
                     std::mt19937       engine(seed_gen());
                     std::shuffle(place_vec_thread.begin(), place_vec_thread.end(), engine);
                     int inner_count = 0;

                     // for each word position
                     for (int itr_place = 0; itr_place < int(place_vec_thread.size()); ++itr_place) {
                        int             place     = place_vec_thread[itr_place];
                        int             word      = doc_word_topic(place, 1);
                        Eigen::MatrixXf x         = embedding_matrix.row(word);
                        int             old_level = doc_word_topic(place, 3);
                        int             old_topic = doc_word_topic(place, 4);

                        ghlda_update_parameters_subtract_thread(old_topic, x, topic2num_thread_2, topic2mu_thread_2, topic2Sigma_thread_2);

                        int hantei_update_doc2level = 0;
                        if (doc2level_vec_thread[doc_thread](old_level, 0) > level_gamma) {
                           doc2level_vec_thread[doc_thread](old_level, 0) = doc2level_vec_thread[doc_thread](old_level, 0) - 1.0;
                           hantei_update_doc2level = 1;
                        }

                        int max_depth_thread = 0;
                        for (int i = 0; i < max_depth_allowed; ++i) {
                           if (doc2level_vec_thread[doc_thread](i, 0) > level_gamma) {
                              max_depth_thread = i + 1;
                           }
                        }

                        if (max_depth_thread > max_depth_allowed) {
                           max_depth_thread = max_depth_allowed;
                        }

                        // Create level topic ratio
                        std::vector <float> level_ratio;
                        float agg_level = 0.0;
                        if (level_allocation_type == 0) {   // Pure LDA
                           for (int i = 0; i < int(new_path_vec.size()); ++i) {
                              float temp = alpha_level_vec(i) + doc2level_vec_thread[doc_thread](i, 0);
                              level_ratio.push_back(temp);
                              agg_level += temp;
                           }
                        }else{   // HLDA
                           for (int i = 0; i < max_depth_thread; ++i) {
                              float first       = 1.0;
                              float first_nume  = hyper_m * hyper_pi + doc2level_vec_thread[doc_thread](i, 0);
                              float first_denom = hyper_pi;
                              for (int j = i; j < max_depth_thread; ++j) {
                                 first_denom += doc2level_vec_thread[doc_thread](j, 0);
                              }
                              first = first_nume / first_denom;
                              float second = 1.0;
                              for (int j = 0; j < i; ++j) {
                                 float second_nume_inside = 0.0;
                                 for (int k = j + 1; k < max_depth_thread; ++k) {
                                    second_nume_inside += doc2level_vec_thread[doc_thread](k, 0);
                                 }
                                 float second_nume         = (1.0 - hyper_m) * hyper_pi + second_nume_inside;
                                 float second_denom_inside = 0.0;
                                 for (int k = j; k < max_depth_thread; ++k) {
                                    second_denom_inside += doc2level_vec_thread[doc_thread](k, 0);
                                 }
                                 float second_denom = hyper_pi + second_denom_inside;
                                 second = second * (second_nume / second_denom);
                              }
                              float temp = first * second;
                              level_ratio.push_back(temp);
                              agg_level += temp;
                           }
                           if (max_depth_thread != max_depth_allowed) {
                              // ADD another level
                              float temp_2 = 1 - agg_level;
                              if (temp_2 < 0.001) {
                                 temp_2 = 0.001;
                              }
                              level_ratio.push_back(temp_2);
                              if (int(level_ratio.size()) < int(new_path_vec.size())) {
                                 level_ratio.push_back(0.001);
                              }
                           }
                        }   //if(level_allocation_type == 0){}else{}

                        float agg_topic = 0;
                        max_val = -9999999999999;
                        std::vector <float> ratio;
                        for (int j = 0; j < int(level_ratio.size()); ++j) {
                           int topic = -1;
                           if (j < int(new_path_vec.size())) {
                              topic = new_path_vec[j];
                           }
                           float log_prob_a = std::log(level_ratio[j]);
                           float log_prob_b = ghlda_posterior_predictive_single_robust(place, j, topic,
                                                                                       nu, kappa,
                                                                                       topic2num_thread_2, topic2mu_thread_2,
                                                                                       topic2Sigma_thread_2, 1, verbose);
                           float log_prob = log_prob_a + log_prob_b;
                           ratio.push_back(log_prob);
                           if (max_val < ratio[j]) {
                              max_val = ratio[j];
                           }
                        }

                        agg_topic = 0;
                        for (int j = 0; j < int(ratio.size()); ++j) {
                           ratio[j]   = ratio[j] - max_val;
                           ratio[j]   = std::exp(ratio[j]);
                           agg_topic += ratio[j];
                        }
                        // sample
                        double prob[ratio.size()];    // Probability array
                        for (int j = 0; j < int(ratio.size()); ++j) {
                           prob[j] = double(ratio[j] / agg_topic);
                        }
                        unsigned int mult_op[int(ratio.size())];
                        int          num_sample = 1;
                        gsl_ran_multinomial(r[thread_id], int(ratio.size()), num_sample, prob, mult_op);
                        int new_level = -1;
                        int new_topic = -1;
                        for (int j = 0; j < int(ratio.size()); ++j) {
                           if (mult_op[j] == 1) {
                              new_level = j;
                              break;
                           }
                        }

                        if (new_level < int(new_path_vec.size())) {
                           new_topic = new_path_vec[new_level];
                        }else{
                           for (auto itr = extended_path_dict.begin(); itr != extended_path_dict.end(); ++itr) {
                              std::vector <int> path_vec = itr->second;
                              if (int(path_vec.size()) == (new_level + 1)) {
                                 int hantei_replace = 1;
                                 for (int k = 0; k < int(new_path_vec.size()); ++k) {
                                    if (new_path_vec[k] != path_vec[k]) {
                                       hantei_replace = 0;
                                    }
                                 }
                                 if (hantei_replace == 1) {
                                    new_path     = itr->first;
                                    new_path_vec = path_vec;
                                    new_topic    = path_vec[new_level];
                                    break;
                                 }
                              }
                           }
                        }

                        if (new_topic != -1) {
                           ghlda_update_parameters_add_thread(new_topic, x, topic2num_thread_2, topic2mu_thread_2, topic2Sigma_thread_2);
                        }

                        if (hantei_update_doc2level == 1) {
                           float temp_val = doc2level_vec_thread[doc_thread](new_level, 0) + 1.0;
                           doc2level_vec_thread[doc_thread](new_level, 0) = temp_val;
                        }
                        std::vector <int> result;
                        result.push_back(place);
                        result.push_back(new_path);
                        result.push_back(new_level);
                        result.push_back(new_topic);
                        new_place_path_level[inner_count] = result;
                        inner_count += 1;
                     }  //for (int itr_place = 0; itr_place < int(place_vec_thread.size()); ++itr_place) {

                     #pragma omp critical
                     {
                        //if(discard_flag == 0){
                        if (1 == 1) {
                           sample_counter += 1.0;
                           int hantei     = 1;
                           int change_doc = -1;
                           for (auto itr2 = new_place_path_level.begin(); itr2 != new_place_path_level.end(); ++itr2) {
                              std::vector <int> result = itr2->second;
                              int change_place         = result[0];
                              change_doc = doc_word_topic(change_place, 0);
                              //int new_path  = result[1];
                              int new_level = result[2];
                              int new_topic = result[3];
                              int old_path  = doc_word_topic(change_place, 2);
                              int old_level = doc_word_topic(change_place, 3);
                              int old_topic = doc_word_topic(change_place, 4);

                              if (1 == 1) {
                                 doc_word_topic(change_place, 2)         = new_path;
                                 doc_word_topic(change_place, 3)         = new_level;
                                 doc_word_topic(change_place, 4)         = new_topic;
                                 doc2level_vec[change_doc](old_level, 0) = doc2level_vec[change_doc](old_level, 0) - 1.0;
                                 doc2level_vec[change_doc](new_level, 0) = doc2level_vec[change_doc](new_level, 0) + 1.0;

                                 if (1 == 1) {
                                    if (hantei == 1) {   // we only need to do this once
                                       hantei = 0;
                                       if (path_dict.find(old_path) != path_dict.end()) {
                                          path2num[old_path] = path2num[old_path] - 1.0;
                                          if (path2num[old_path] < 0.1) {
                                             std::map <int, std::vector <int> > path_dict_t;
                                             std::unordered_map <int, float>    path2num_t;
                                             for (auto itr4 = path2num.begin(); itr4 != path2num.end(); ++itr4) {
                                                int path_num_t = itr4->first;
                                                if (path2num[path_num_t] > 0.1) {
                                                   path2num_t[path_num_t]  = itr4->second;
                                                   path_dict_t[path_num_t] = path_dict[path_num_t];
                                                }
                                             }
                                             path2num.clear();
                                             path2num = path2num_t;
                                             path_dict.clear();
                                             path_dict = path_dict_t;
                                             // OLD VERSION
                                             //path_dict.erase(old_path);
                                             //path2num.erase(old_path);
                                          }
                                       }
                                       if (path_dict.find(new_path) != path_dict.end()) {
                                          path2num[new_path] = path2num[new_path] + 1.0;
                                       }else{   // Completely new path
                                          path_dict[new_path] = new_path_vec;
                                          path2num[new_path]  = 1.0;
                                          for (int k = 0; k < int(new_path_vec.size()); ++k) {
                                             if (topic2mu.find(new_path_vec[k]) == topic2mu.end()) {
                                                topic2mu[new_path_vec[k]] = zero_mu;
                                             }
                                             if (topic2Sigma.find(new_path_vec[k]) == topic2Sigma.end()) {
                                                topic2Sigma[new_path_vec[k]] = zero_Sigma;
                                             }
                                             if (topic2num.find(new_path_vec[k]) == topic2num.end()) {
                                                topic2num[new_path_vec[k]] = 0.0;
                                             }
                                          }
                                       }
                                       doc2path[change_doc] = new_path;
                                       update_num_path_num_topics(path_dict);
                                    }   //if(hantei==1){
                                    int             change_word = doc_word_topic(change_place, 1);
                                    Eigen::MatrixXf change_x    = embedding_matrix.row(change_word);
                                    ghlda_update_parameters_subtract(old_topic, change_x);
                                    ghlda_update_parameters_add(new_topic, change_x);
                                 }//if (1 == 1)
                              }    //if (path2num[old_path] > 0){
                           }       //for(auto itr2=new_place_path_level.begin()
                        }          //if(discard_flag == 0){
                     }             //#pragma omp critical
                  } //if (stop_hantei != 1) {
               } //if (1 == 1) {
            } //if (stop_hantei != 1) {
         }    //for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
      }       //for (int itr_outer = 0; itr_outer < num_iteration; ++itr_outer) {

      // free memory
      for (int i = 0; i < num_threads; i++) {
         gsl_rng_free(r[i]);
      }
   }//ghlda_ncrp_collapsed_gibbs_sample_parallel

   void ghlda_ncrp_create_topic2word_prob(
      std::unordered_map <int, Eigen::MatrixXf>& topic2word_prob,
      std::unordered_map <int, Eigen::MatrixXf>& topic2mu_thread,
      std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_thread,
      float eta_thres, std::unordered_map <int, int>& update_topic_thread) {

      // create topic2level
      std::unordered_map <int, int> topic2level;
      for (auto itr = path_dict.begin(); itr != path_dict.end(); ++itr) {
         std::vector <int> path_vec = itr->second;
         for (int j = 0; j < int(path_vec.size()); ++j) {
            topic2level[path_vec[j]] = j;
         }
      }

      int size = 0;
      for (auto itr = id2word.begin(); itr != id2word.end(); ++itr) {
         size += 1;
      }
      for (auto itr = topic2mu.begin(); itr != topic2mu.end(); ++itr) {
         int topic = itr->first;

         std::cout << "topic: " << topic << std::endl;

         if (update_topic_thread.find(topic) != update_topic_thread.end()) {
            Eigen::MatrixXf word_prob    = Eigen::MatrixXf::Zero(size, 1);
            float           max_val      = -999999999999;
            Eigen::MatrixXf Sigma_stable = topic2Sigma_thread[topic] + level2Psi[topic2level[topic]];
            for (auto itr2 = id2word.begin(); itr2 != id2word.end(); ++itr2) {
               int             word = itr2->first;
               Eigen::MatrixXf x    = embedding_matrix.row(word);
               float           prob = prob_mass_mvt(x, nu, topic2mu_thread[topic], Sigma_stable, 1, 0);
               if (topic == -1) {
                  std::cout << "topic -1:" << prob << std::endl;
               }
               if (max_val < prob) {
                  max_val = prob;
               }
               word_prob(word, 0) = prob;
            }
            float agg = 0.0;
            for (auto itr2 = id2word.begin(); itr2 != id2word.end(); ++itr2) {
               int word = itr2->first;
               word_prob(word, 0) = std::exp(word_prob(word, 0) - max_val) + eta_thres;
               agg += word_prob(word, 0);
            }
            for (auto itr2 = id2word.begin(); itr2 != id2word.end(); ++itr2) {
               int word = itr2->first;
               word_prob(word, 0) = word_prob(word, 0) / agg;
            }
            topic2word_prob[topic] = word_prob;
         }
      }
   }

   float ghlda_ncrp_posterior_predictive_robust_test(const Eigen::MatrixXi& doc_word_topic_thread_temp,
                                                     const std::vector <int>& topic_vec,
                                                     const int& level,
                                                     const int& last_topic,
                                                     const float& nu, const float& kappa,
                                                     std::unordered_map <int, float>& topic2num_temp,
                                                     std::unordered_map <int, Eigen::MatrixXf>& topic2mu_temp,
                                                     std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_temp,
                                                     std::unordered_map <int, float> topic2third_nume_inside,
                                                     const int& log_true, const int& verbose) {
      std::vector <int> place_level_vec;
      for (int i = 0; i < int(doc_word_topic_thread_temp.rows()); ++i) {
         if (last_topic != 1) {
            if (doc_word_topic_thread_temp(i, 3) == level) {
               place_level_vec.push_back(i);
            }
         }else{
            if (doc_word_topic_thread_temp(i, 3) >= level) {
               place_level_vec.push_back(i);
            }
         }
      }

      const int topic = topic_vec[level];
      float     n     = 0;
      float     m     = 0;
      n = float(topic2num_temp[topic]);
      m = float(place_level_vec.size());

      // Actually we have to re calculate mu and sigma
      float       log_prob           = 0.0;
      float       prob               = 0.0;
      float       first              = 0.0;
      float       second_nume        = 0.0;
      float       second_denom       = 0.0;
      float       third_nume_inside  = 0.0;
      float       third_denom_inside = 0.0;
      float       third_nume         = 0.0;
      float       third_denom        = 0.0;
      float       fourth_nume        = 0.0;
      float       fourth_denom       = 0.0;
      std::string method             = "llt";

      first = 0 - std::log(float( int(doc_word_topic_thread_temp.rows())) * embedding_dimension / 2.0 * std::log(M_PI));
      if (verbose == 1) {
         std::cout << "  first " << first << std::endl;
      }

      for (int d = 0; d < embedding_dimension; ++d) {
         second_nume  = second_nume + std::lgamma((nu + n + m - float(d)) / 2.0);
         second_denom = second_denom + std::lgamma((nu + n - float(d)) / 2.0);
      }

      if (verbose == 1) {
         std::cout << "  second_nume " << second_nume << " " << std::to_string(nu) << std::endl;
         std::cout << "  second_denom " << second_denom << " " << std::to_string(nu) << std::endl;
      }

      std::unordered_map <int, float> thres2third_nume_inside;

      if (1 == 1) {

        Eigen::MatrixXf mu;
        Eigen::MatrixXf Sigma;
         if (topic2mu_temp.find(topic) == topic2mu_temp.end()) {
            //std::cout << "Mu topic " << topic << " does not exist" << std::endl;
            mu = zero_mu;
         }else{
            mu           = topic2mu_temp[topic];
         }
         if (topic2Sigma_temp.find(topic) == topic2Sigma_temp.end()) {
            //std::cout << "Sigma topic " << topic << " does not exist" << std::endl;
            Sigma = zero_Sigma;
         }else{
            Sigma        = topic2Sigma_temp[topic];
         }


         Eigen::MatrixXf Sigma_stable = Psi + Sigma + ((kappa * n) / (kappa + n)) * (mu - embedding_center).transpose() * (mu - embedding_center);
         Sigma_stable = rescale * Sigma_stable;

         if (method == "llt") {
            Eigen::LLT <Eigen::MatrixXf> llt;
            llt.compute(Sigma_stable);
            auto& U = llt.matrixL();
            for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
               if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
                  third_nume_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
               }else{
                  third_nume_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
               }
               thres2third_nume_inside[i + 1] = third_nume_inside;
            }
         }        //if(method == "llt"){
         if (verbose == 1) {
            std::cout << "third_nume_inside " << third_nume_inside << std::endl;
         }
         topic2third_nume_inside[topic] = third_nume_inside;
      }else{
         third_nume_inside = topic2third_nume_inside[topic];
      }

      // third_denom
      //float           N_temp     = topic2num_temp[topic];
      //Eigen::MatrixXf mu_temp    = topic2mu_temp[topic];
      //Eigen::MatrixXf Sigma_temp = topic2Sigma_temp[topic];

      float           N_temp = 0.0;
      Eigen::MatrixXf mu_temp;
      Eigen::MatrixXf Sigma_temp;
      if (topic2mu_temp.find(topic) == topic2mu_temp.end()) {
         //std::cout << "Mu topic " << topic << " does not exist" << std::endl;
         mu_temp = zero_mu;
      }else{
         mu_temp           = topic2mu_temp[topic];
         N_temp     = topic2num_temp[topic];
      }
      if (topic2Sigma_temp.find(topic) == topic2Sigma_temp.end()) {
         //std::cout << "Sigma topic " << topic << " does not exist" << std::endl;
         Sigma_temp = zero_Sigma;
      }else{
         Sigma_temp        = topic2Sigma_temp[topic];
         N_temp     = topic2num_temp[topic];
      }


      int hantei = 0;
      if (N_temp == 0) {
         hantei = 1;
      }

      for (int i = 0; i < int(place_level_vec.size()); ++i) {
         int             alter_word = doc_word_topic_thread_temp(place_level_vec[i], 1);
         Eigen::MatrixXf alter_x    = embedding_matrix.row(alter_word);
         if (hantei == 0) {
            ghlda_update_mu_sigma(topic, alter_x, N_temp, mu_temp, Sigma_temp);
         }else{
            N_temp += 1;
            mu_temp = mu_temp + alter_x;
         }
      }

      if (hantei == 1 && N_temp > 0) {
         mu_temp = mu_temp / N_temp;
      }

      float value  = (kappa * n + m);
      float value2 = (kappa + n + m);
      float value3 = ((kappa * n + m) / (kappa + n + m));

      Eigen::MatrixXf Sigma_stable = Psi + Sigma_temp + ((kappa * n + m) / (kappa + n + m)) * (mu_temp - embedding_center).transpose() * (mu_temp - embedding_center);
      Sigma_stable = rescale * Sigma_stable;
      if (method == "llt") {
         Eigen::LLT <Eigen::MatrixXf> llt;
         llt.compute(Sigma_stable);
         auto& U = llt.matrixL();
         for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
            if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
               third_denom_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
            }else{
               third_denom_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
            }
         }
         third_nume  = ((nu + n) / 2.0) * third_nume_inside;
         third_denom = ((nu + m + n) / 2.0) * third_denom_inside;
      }   //if(method == "llt"){

      if (verbose == 1) {
         std::cout << "  value " << value << std::endl;
         std::cout << "  value2 " << value2 << std::endl;
         std::cout << "  value3 " << value3 << std::endl;
         std::cout << "  third_nume " << third_nume << std::endl;
         std::cout << "  third_denom " << third_denom << std::endl;
      }

      fourth_nume  = (embedding_dimension / 2.0) * std::log(kappa + n);
      fourth_denom = (embedding_dimension / 2.0) * std::log(kappa + n + m);

      if (verbose == 1) {
         std::cout << "  n " << n << std::endl;
         std::cout << "  m " << m << std::endl;
         std::cout << "  fourth_nume " << fourth_nume << std::endl;
         std::cout << "  fourth_denom " << fourth_denom << std::endl;
      }

      log_prob = first + second_nume - second_denom + third_nume - third_denom + fourth_nume - fourth_denom;
      log_prob = log_prob - inner_rescale;

      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      return(prob);
   }

   float ghlda_posterior_predictive_single_robust_test(const Eigen::MatrixXi& doc_word_topic_thread_temp,
                                                       const int& place,
                                                       const int& level,
                                                       const int& topic,
                                                       const float& nu, const float& kappa,
                                                       std::unordered_map <int, float>& topic2num_temp,
                                                       std::unordered_map <int, Eigen::MatrixXf>& topic2mu_temp,
                                                       std::unordered_map <int, Eigen::MatrixXf>& topic2Sigma_temp,
                                                       const int& log_true, const int& verbose) {
      Eigen::MatrixXf Psi_t = level2Psi[level];

      //std::vector<int> place_vec = doc2place_vec[doc];
      std::vector <int> place_level_vec;
      place_level_vec.push_back(place);

      // KOKOGA MONDAI
      float n = 0;
      float m = 0;
      n = float(topic2num_temp[topic]);   // - float(place_vec.size());
      m = float(place_level_vec.size());

      // Actually we have to re calculate mu and sigma
      float log_prob           = 0.0;
      float prob               = 0.0;
      float first              = 0.0;
      float second_nume        = 0.0;
      float second_denom       = 0.0;
      float third_nume_inside  = 0.0;
      float third_denom_inside = 0.0;
      float third_nume         = 0.0;
      float third_denom        = 0.0;
      float fourth_nume        = 0.0;
      float fourth_denom       = 0.0;
      // Choleskey fast
      std::string method = "llt";

      // KOKO
      first = 0 - std::log(float(place_level_vec.size()) * embedding_dimension / 2.0 * std::log(M_PI));
      if (verbose == 1) {
         std::cout << "  first " << first << std::endl;
      }

      for (int d = 0; d < embedding_dimension; ++d) {
         second_nume  = second_nume + std::lgamma((nu + n + m - float(d)) / 2.0);
         second_denom = second_denom + std::lgamma((nu + n - float(d)) / 2.0);
      }

      //second_nume = std::lgamma((nu + n + m + float(embedding_dimension)) / 2.0);
      //second_denom = std::lgamma((nu + n + float(embedding_dimension)) / 2.0);

      if (verbose == 1) {
         std::cout << "  second_nume " << second_nume << std::endl;
         std::cout << "  second_denom " << second_denom << std::endl;
      }

      std::unordered_map <int, float> thres2third_nume_inside;

      if (1 == 1) {
         //if(topic2third_nume_inside.find(topic)==topic2third_nume_inside.end()){
         Eigen::MatrixXf mu;
         Eigen::MatrixXf Sigma;
          if (topic2mu_temp.find(topic) == topic2mu_temp.end()) {
             //std::cout << "Mu topic " << topic << " does not exist" << std::endl;
             mu = zero_mu;
          }else{
             mu           = topic2mu_temp[topic];
          }
          if (topic2Sigma_temp.find(topic) == topic2Sigma_temp.end()) {
             //std::cout << "Sigma topic " << topic << " does not exist" << std::endl;
             Sigma = zero_Sigma;
          }else{
             Sigma        = topic2Sigma_temp[topic];
          }
         Eigen::MatrixXf Sigma_stable = Psi_t + Sigma + ((kappa * n) / (kappa + n)) * (mu - embedding_center).transpose() * (mu - embedding_center);
         Sigma_stable = rescale * Sigma_stable;
         if (method == "llt") {
            Eigen::LLT <Eigen::MatrixXf> llt;
            llt.compute(Sigma_stable);
            auto& U = llt.matrixL();
            for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
               if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
                  third_nume_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
               }else{
                  third_nume_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
               }
               thres2third_nume_inside[i + 1] = third_nume_inside;
            }
         }        //if(method == "llt"){
         if (verbose == 1) {
            std::cout << "third_nume_inside " << third_nume_inside << std::endl;
         }
      }else{
      }

      //third_nume = ((nu + n) / 2.0) * third_nume_inside;
      // third_denom
      //float           N_temp     = topic2num_temp[topic];
      //Eigen::MatrixXf mu_temp    = topic2mu_temp[topic];
      //Eigen::MatrixXf Sigma_temp = topic2Sigma_temp[topic];

      float           N_temp = 0.0;
      Eigen::MatrixXf mu_temp;
      Eigen::MatrixXf Sigma_temp;
      if (topic2mu_temp.find(topic) == topic2mu_temp.end()) {
         //std::cout << "Mu topic " << topic << " does not exist" << std::endl;
         mu_temp = zero_mu;
      }else{
         mu_temp           = topic2mu_temp[topic];
         N_temp     = topic2num_temp[topic];
      }
      if (topic2Sigma_temp.find(topic) == topic2Sigma_temp.end()) {
         //std::cout << "Sigma topic " << topic << " does not exist" << std::endl;
         Sigma_temp = zero_Sigma;
      }else{
         Sigma_temp        = topic2Sigma_temp[topic];
         N_temp     = topic2num_temp[topic];
      }




      int hantei = 0;
      if (N_temp == 0) {
         hantei = 1;
      }

      for (int i = 0; i < int(place_level_vec.size()); ++i) {
         int             alter_word = doc_word_topic_thread_temp(place_level_vec[i], 1);
         Eigen::MatrixXf alter_x    = embedding_matrix.row(alter_word);
         if (hantei == 0) {
            ghlda_update_mu_sigma(topic, alter_x, N_temp, mu_temp, Sigma_temp);
         }else{
            N_temp += 1;
            mu_temp = mu_temp + alter_x;
         }
      }

      if (hantei == 1 && N_temp > 0) {
         mu_temp = mu_temp / N_temp;
      }

      float value  = (kappa * n + m);
      float value2 = (kappa + n + m);
      float value3 = ((kappa * n + m) / (kappa + n + m));

      Eigen::MatrixXf Sigma_stable = Psi_t + Sigma_temp + ((kappa * n + m) / (kappa + n + m)) * (mu_temp - embedding_center).transpose() * (mu_temp - embedding_center);
      Sigma_stable = rescale * Sigma_stable;
      if (method == "llt") {
         Eigen::LLT <Eigen::MatrixXf> llt;
         llt.compute(Sigma_stable);
         auto& U = llt.matrixL();
         for (int i = 0; i < int(Sigma_stable.rows()); ++i) {
            if (min_diagonal_val != 0.0 && U(i, i) < min_diagonal_val) {
               third_denom_inside += 2 * (std::log(min_diagonal_val)) - std::log(rescale);
            }else{
               third_denom_inside += 2 * (std::log(U(i, i))) - std::log(rescale);
            }
         }
         third_nume  = ((nu + n) / 2.0) * third_nume_inside;
         third_denom = ((nu + m + n) / 2.0) * third_denom_inside;
      }

      if (verbose == 1) {
         std::cout << "  value " << value << std::endl;
         std::cout << "  value2 " << value2 << std::endl;
         std::cout << "  value3 " << value3 << std::endl;
         std::cout << "  third_nume " << third_nume << std::endl;
         std::cout << "  third_denom " << third_denom << std::endl;
      }

      fourth_nume  = (embedding_dimension / 2.0) * std::log(kappa + n);
      fourth_denom = (embedding_dimension / 2.0) * std::log(kappa + n + m);

      if (verbose == 1) {
         std::cout << "  n " << n << std::endl;
         std::cout << "  m " << m << std::endl;
         std::cout << "  fourth_nume " << fourth_nume << std::endl;
         std::cout << "  fourth_denom " << fourth_denom << std::endl;
      }

      log_prob = first + second_nume - second_denom + third_nume - third_denom + fourth_nume - fourth_denom;
      log_prob = log_prob - inner_rescale;
      if (log_true == 1) {
         prob = log_prob;
      }else{
         prob = std::exp(log_prob);
      }
      return(prob);
   }

   void ghlda_ncrp_evaluate_held_out_log_likelihood(const std::vector <int> doc_test_ids, const int num_iteration, const int burn_in, const float path_thres, const int approx, int num_threads, int verbose) {

      if (verbose == 1) {
         std::cout << "Enter glda_evaluate_held_out_log_likelihood" << std::endl;
      }

      if (num_threads < 1) {
         num_threads = omp_get_max_threads();
         std::cout << "Using " << num_threads << " threads" << std::endl;
      }

      omp_set_num_threads(num_threads);
      int max_num_threads = omp_get_max_threads();
      const gsl_rng_type *T;
      gsl_rng **          r;
      gsl_rng_env_setup();
      T = gsl_rng_mt19937;
      r = (gsl_rng **)malloc(max_num_threads * sizeof(gsl_rng *));
      std::random_device rand_device;
      std::mt19937       mt(rand_device());
      for (int i = 0; i < max_num_threads; i++) {
         r[i] = gsl_rng_alloc(T);
         gsl_rng_set(r[i], mt());
      }

      // create_topic2word_prob: Initially we have to update everything
      std::unordered_map <int, int> update_topic;
      for (auto itr = topic2num.begin(); itr != topic2num.end(); ++itr) {
         update_topic[itr->first] = 1;
      }
      // version 1
      //ghlda_ncrp_create_topic2word_prob(topic2word_prob, topic2mu, topic2Sigma, eta_thres, update_topic);
      // version 2
      create_topic2word_map();

      if (verbose == 1) {
         std::cout << "Before pragma omp parallel for" << std::endl;
      }

      float counter = 0.0;
      #pragma omp parallel for
      for (int k = 0; k < int(doc_test_ids.size()); ++k) {
         // version 1
         std::unordered_map <int, Eigen::MatrixXf> topic2word_prob_thread = topic2word_prob;
         // version 2
         std::unordered_map <int, std::unordered_map <int, float> > topic2word_map_thread = topic2word_map;
         std::unordered_map <int, int>             update_topic_thread;
         int             doc_test = doc_test_ids[k];
         int             doc_size = int(doc2place_vec_test[doc_test].size());
         Eigen::MatrixXi doc_word_topic_thread = Eigen::MatrixXi::Zero(doc_size, 8);  // take log sum of this
         for (int i = 0; i < doc_size; ++i) {
            int place_t = doc2place_vec_test[doc_test][i];
            doc_word_topic_thread(i, 0) = doc_word_topic_test(place_t, 0);
            doc_word_topic_thread(i, 1) = doc_word_topic_test(place_t, 1);
            doc_word_topic_thread(i, 2) = doc_word_topic_test(place_t, 2);
            doc_word_topic_thread(i, 3) = doc_word_topic_test(place_t, 3);
         }

         if (verbose == 1) {
            std::cout << "Before for (int up_to = 0" << std::endl;
         }

         // for every word position
         Eigen::MatrixXf heldout_prob = Eigen::MatrixXf::Zero(doc_word_topic_thread.rows(), 1);
         std::unordered_map <int, std::unordered_map <int, float> > place2level_weight_thread;
         for (int up_to = 0; up_to < int(doc_word_topic_thread.rows()); ++up_to) {
            int thread_id = omp_get_thread_num();
            std::unordered_map <int, float>           topic2num_thread   = topic2num;
            std::unordered_map <int, Eigen::MatrixXf> topic2mu_thread    = topic2mu;
            std::unordered_map <int, Eigen::MatrixXf> topic2Sigma_thread = topic2Sigma;
            std::unordered_map <int, float>           path2num_thread    = path2num;
            Eigen::MatrixXf level_vec_sub = Eigen::MatrixXf::Zero(num_topics, 1);
            Eigen::MatrixXi doc_word_topic_thread_temp = Eigen::MatrixXi::Zero(up_to + 1, 10);

            for (int i = 0; i < (up_to + 1); ++i) {
               doc_word_topic_thread_temp(i, 0) = doc_word_topic_thread(i, 0);
               doc_word_topic_thread_temp(i, 1) = doc_word_topic_thread(i, 1);
               doc_word_topic_thread_temp(i, 2) = doc_word_topic_thread(i, 2);
               doc_word_topic_thread_temp(i, 3) = doc_word_topic_thread(i, 3);
               int level = doc_word_topic_thread_temp(i, 3);
               if (i == up_to) {
                  level_vec_sub(level, 0) += 1.0;
               }else{
                  for (auto itr3 = place2level_weight_thread[i].begin(); itr3 != place2level_weight_thread[i].end(); ++itr3) {
                     level = itr3->first;
                     float val = itr3->second;
                     level_vec_sub(level, 0) += val;
                  }
               } //for (auto itr3 = place2topic_weight_thread[i].begin()
            }    //or (int i = 0; i < (up_to + 1); ++i) {

            float agg   = 0;
            float count = 0;
            update_topic_thread.clear();
            std::unordered_map <int, float> level_weight_thread;
            for (int rep = 0; rep < num_iteration; ++rep) {
               // Step A: sample path for a document
               float                           agg_topic = 0;
               float                           max_val   = -9999999999999;
               std::vector <int>               path_id_vec;
               std::vector <float>             ratio;
               std::vector <std::string>       topic_string_vec;
               std::unordered_map <int, float> topic2third_nume_inside;
               for (auto itr = path_dict.begin(); itr != path_dict.end(); ++itr) {
                  int path_id = itr->first;
                  path_id_vec.push_back(path_id);
                  std::vector <int> topic_vec = itr->second;
                  float             temp      = 0.0;
                  float             temp_in   = 0.0;
                  for (int j = 0; j < int(topic_vec.size()); ++j) {
                     int last_hantei = 0;
                     if (j == int(topic_vec.size()) - 1) {
                        last_hantei = 1;
                     }
                     temp_in = ghlda_ncrp_posterior_predictive_robust_test(doc_word_topic_thread_temp,
                                                                           topic_vec, j, last_hantei, nu, kappa, topic2num_thread, topic2mu_thread,
                                                                           topic2Sigma_thread, topic2third_nume_inside, 1, 0);
                     temp += temp_in;
                  }
                  if (path2num_thread.find(path_id) != path2num_thread.end()) {
                     temp += std::log(path2num_thread[path_id] + path_gamma);
                  }else{
                     temp += std::log(path_gamma);
                  }
                  ratio.push_back(temp);
                  if (max_val < temp) {
                     max_val = temp;
                  }
               } //for (auto itr = path_dict.begin(); itr != path_dict.end(); ++itr) {
               // sample
               for (int i = 0; i < int(ratio.size()); ++i) {
                  ratio[i]   = ratio[i] - max_val;
                  ratio[i]   = std::exp(ratio[i]);
                  agg_topic += ratio[i];
               }
               double prob_path[int(ratio.size())];   // Probability array
               for (int i = 0; i < int(ratio.size()); ++i) {
                  prob_path[i] = double(ratio[i] / agg_topic);
               }

               unsigned int mult_op[ratio.size()];
               int          num_sample = 1;
               gsl_ran_multinomial(r[thread_id], int(ratio.size()), num_sample, prob_path, mult_op);
               int new_path = -1;
               std::vector<int> path_id_vec_2;
               std::vector<float> prob_path_2;
               for (int i = 0; i < int(ratio.size()); ++i) {
                  if(prob_path[i] > path_thres){
                      prob_path_2.push_back(float(prob_path[i]));
                      path_id_vec_2.push_back(path_id_vec[i]);
                  }
                  if (mult_op[i] == 1) {
                     new_path = path_id_vec[i];
                  }
               }

               for (int kkk = 0; kkk < int(prob_path_2.size()); ++kkk) {
                   new_path = path_id_vec_2[kkk];

                   // new_path
                   std::vector <int> new_path_vec;
                   new_path_vec = path_dict[new_path];
                   update_topic_thread.clear();
                   std::unordered_map <int, float> topic_weight_thread;
                   int             place     = int(doc_word_topic_thread_temp.rows()) - 1;
                   int             word      = doc_word_topic_thread_temp(place, 1);
                   int             old_level = doc_word_topic_thread_temp(place, 3);
                   Eigen::MatrixXf x         = embedding_matrix.row(word);

                   // Create level topic ratio
                   std::vector <float> level_ratio;
                   float agg_level        = 0.0;
                   int   max_depth_thread = max_depth_allowed;
                   if (level_allocation_type == 0) {    // Pure LDA
                      for (int i = 0; i < int(new_path_vec.size()); ++i) {
                         float temp = alpha_level_vec(i) + level_vec_sub(i, 0);
                         level_ratio.push_back(temp);
                         agg_level += temp;
                      }
                   }else{    // HLDA
                      for (int i = 0; i < max_depth_thread; ++i) {
                         float first       = 1.0;
                         float first_nume  = hyper_m * hyper_pi + level_vec_sub(i, 0);
                         float first_denom = hyper_pi;
                         for (int j = i; j < max_depth_thread; ++j) {
                            first_denom += level_vec_sub(j, 0);
                         }
                         first = first_nume / first_denom;
                         float second = 1.0;
                         for (int j = 0; j < i; ++j) {
                            float second_nume_inside = 0.0;
                            for (int k = j + 1; k < max_depth_thread; ++k) {
                               second_nume_inside += level_vec_sub(k, 0);
                            }
                            float second_nume         = (1.0 - hyper_m) * hyper_pi + second_nume_inside;
                            float second_denom_inside = 0.0;
                            for (int k = j; k < max_depth_thread; ++k) {
                               second_denom_inside += level_vec_sub(k, 0);
                            }
                            float second_denom = hyper_pi + second_denom_inside;
                            second = second * (second_nume / second_denom);
                         }
                         float temp = first * second;
                         level_ratio.push_back(temp);
                         agg_level += temp;
                      }
                   }    //if(level_allocation_type == 0){}else{}

                   max_val = -9999999999999;
                   ratio.clear();
                   for (int j = 0; j < int(level_ratio.size()); ++j) {
                      int topic = -1;
                      if (j < int(new_path_vec.size())) {
                         topic = new_path_vec[j];
                      }
                      float log_prob_a = std::log(level_ratio[j]);
                      float log_prob_b = ghlda_posterior_predictive_single_robust_test(doc_word_topic_thread_temp,
                                                                                       place, j, topic, nu, kappa,
                                                                                       topic2num_thread, topic2mu_thread,
                                                                                       topic2Sigma_thread, 1, verbose);
                      float log_prob = log_prob_a + log_prob_b;
                      ratio.push_back(log_prob);
                      if (max_val < ratio[j]) {
                         max_val = ratio[j];
                      }
                   } //for (int j = 0; j < int(level_ratio.size()); ++j) {

                   agg_topic = 0;
                   for (int j = 0; j < int(ratio.size()); ++j) {
                      ratio[j]   = ratio[j] - max_val;
                      ratio[j]   = std::exp(ratio[j]);
                      agg_topic += ratio[j];
                   }
                   // sample
                   double prob_level[ratio.size()];     // Probability array
                   for (int j = 0; j < int(ratio.size()); ++j) {
                      prob_level[j] = double(ratio[j] / agg_topic);
                   }
                   unsigned int mult_op_level[int(ratio.size())];
                   num_sample = 1;
                   gsl_ran_multinomial(r[thread_id], int(ratio.size()), num_sample, prob_level, mult_op_level);
                   int new_level = -1;
                   int new_topic = -1;
                   for (int j = 0; j < int(ratio.size()); ++j) {
                      if (mult_op_level[j] == 1) {
                         new_level = j;
                         break;
                      }
                   }
                   level_vec_sub(old_level, 0)         -= 1.0;
                   level_vec_sub(new_level, 0)         += 1.0;
                   doc_word_topic_thread_temp(place, 3) = new_level;
                   doc_word_topic_thread(place, 3)      = new_level;
                   new_topic = path_dict[new_path][new_level];
                   if (rep > burn_in) {
                      update_topic_thread.clear();
                      for (int j = 0; j < int(ratio.size()); ++j) {
                          new_level = j;
                          new_topic = path_dict[new_path][new_level];
                          // version 1
                          //agg   += float(prob_level[j]) * topic2word_prob_thread[new_topic](word, 0);
                          // version 2
                          if(topic2word_map_thread[new_topic].find(word)!=topic2word_map_thread[new_topic].end()){
                            agg += float(prob_path_2[kkk])*float(prob_level[j]) * (topic2word_map_thread[new_topic][word] + beta) / (voc * beta + topic2num_thread[new_topic]);
                          }else{
                            agg += float(prob_path_2[kkk])*float(prob_level[j]) * beta / (voc * beta + topic2num_thread[new_topic]);
                          }
                      }
                      count += float(prob_path_2[kkk]);
                      if (level_weight_thread.find(new_level) == level_weight_thread.end()) {
                         level_weight_thread[new_level] = 1.0;
                      }else{
                         level_weight_thread[new_level] += 1.0;
                      }
                   } //if (rep > burn_in) {
                }//for (int i = 0; i < int(ratio.size()); ++i) {
            }    //for (int rep = 0; rep < num_iteration; ++rep) {

            float prob_2 = agg / count;
            heldout_prob(up_to, 0) = prob_2;

            // Update place2topic_weight_thread
            float agg2 = 0.0;
            for (auto itr3 = level_weight_thread.begin(); itr3 != level_weight_thread.end(); ++itr3) {
               agg2 += 1.0;
            }
            for (auto itr3 = level_weight_thread.begin(); itr3 != level_weight_thread.end(); ++itr3) {
               itr3->second = itr3->second / agg2;
            }
            place2level_weight_thread[up_to] = level_weight_thread;
         }//for (int up_to = 0; up_to < int(doc_word_topic_thread.rows()); ++up_to) {
         #pragma omp critical
         {
            doc_test2heldout_prob[doc_test] = heldout_prob;
            counter += 1;
            std::cout << "Finished: " << counter / float(doc_test_ids.size()) << std::endl;
         } //#pragma omp critical
      }    //for (int k = 0; k < int(doc_test_ids.size()); ++k)
   }

   // GHLDA-ncrp END //

   // CGTM //
   std::unordered_map <long long, Eigen::MatrixXf> doc2eta_vec,doc2eta_vec_test;
   std::unordered_map <long long, Eigen::MatrixXf> doc2lambda_vec,doc2lambda_vec_test;
   Eigen::MatrixXf initial_eta_vec, initial_lambda_vec;
   Eigen::MatrixXf mu_for_topicprop_prior, Sigma_for_topicprop_prior, Psi_for_topicprop_prior;
   float kappa_for_topicprop_prior;
   Eigen::MatrixXf mu_c, Sigma_c;
   Eigen::MatrixXf doc2eta_table, doc2lambda_table;
   int max_try = 100;

   // Polya-Gamma CODES ARE ADOPTED FROM https://cran.r-project.org/web/packages/pgdraw/index.html
   // Author:	Daniel F. Schmidt, Enes Makalic
   float rand_unif_thread(gsl_rng *r_thread) {
      float value = gsl_rng_uniform(r_thread);

      return(value);
   }

   float rand_norm_thread(gsl_rng *r_thread, float mu, float sigma) {
      float value = (float)gsl_ran_gaussian(r_thread, (double)sigma);

      return(value);
   }

   float exprnd_thread(gsl_rng *r_thread, float mu) {
      float value = rand_unif_thread(r_thread);

      return(-mu * (float)std::log(1.0 - value));
   }

   // Sample truncated gamma random variates
   float truncgamma_thread(gsl_rng *r_thread) {
      float c = MATH_PI_2;
      float X, gX;
      bool  done = false;

      while (!done) {
         X  = exprnd_thread(r_thread, 1.0) * 2.0 + c;
         gX = MATH_SQRT_PI_2 / (float)std::sqrt(X);
         float value = rand_unif_thread(r_thread);
         if (value <= gX) {
            done = true;
         }
      }// while(!done){
      return(X);
   }

   // Generate inverse gaussian random variates
   float randinvg_thread(gsl_rng *r_thread, float mu) {
      // sampling
      float u   = rand_norm_thread(r_thread, 0.0, 1.0);
      float V   = u * u;
      float out = mu + 0.5 * mu * (mu * V - (float)std::sqrt(4.0 * mu * V + mu * mu * V * V));

      if (rand_norm_thread(r_thread, 0.0, 1.0) > mu / (mu + out)) {
         out = mu * mu / out;
      }
      return(out);
   }

   // Sample truncated inverse Gaussian random variates
   float tinvgauss_thread(gsl_rng *r_thread, float z, float t) {
      float X, u;
      float mu = 1.0 / z;

      // Pick sampler
      if (mu > t) {
         // Sampler based on truncated gamma
         // Algorithm 3 in the Windle (2013) PhD thesis, page 128
         while (1) {
            //u = rand_unif(0.0, 1.0);
            u = rand_unif_thread(r_thread);
            X = 1.0 / truncgamma_thread(r_thread);
            if ((float)std::log(u) < (-z * z * 0.5 * X)) {
               break;
            }
         }
      }else{
         // Rejection sampler
         X = t + 1.0;
         while (X >= t) {
            X = randinvg_thread(r_thread, mu);
         }
      }
      return(X);
   }

   float aterm(int n, float x, float t) {
      float f = 0;

      if (x <= t) {
         f = MATH_LOG_PI + (float)std::log(n + 0.5) + 1.5 * (MATH_LOG_2_PI - (float)std::log(x)) - 2 * (n + 0.5) * (n + 0.5) / x;
      } else {
         f = MATH_LOG_PI + (float)std::log(n + 0.5) - x * MATH_PI2_2 * (n + 0.5) * (n + 0.5);
      }
      return((float)std::exp(f));
   }

   // Sample PG(1,z)
   float samplepg_thread(gsl_rng *r_thread, float z) {
      //  PG(b, z) = 0.25 * J*(b, z/2)
      z = (float)std::fabs((float)z) * 0.5;

      // Point on the intersection IL = [0, 4/ log 3] and IR = [(log 3)/pi^2, \infty)
      float t = MATH_2_PI;

      // Compute p, q and the ratio q / (q + p)
      // (derived from scratch; derivation is not in the original paper)
      float K    = z * z / 2.0 + MATH_PI2 / 8.0;
      float logA = (float)std::log(4.0) - MATH_LOG_PI - z;
      float logK = (float)std::log(K);
      float Kt   = K * t;
      float w    = (float)std::sqrt(MATH_PI_2);

      // KOKO R::pnorm
      //double logf1 = logA + R::pnorm(w*(t*z - 1),0.0,1.0,1,1) + logK + Kt;
      float logf1 = logA + (float)std::log(gsl_cdf_gaussian_P((double)w * (t * z - 1), 1.0)) + logK + Kt;
      //double logf2 = logA + 2*z + R::pnorm(-w*(t*z+1),0.0,1.0,1,1) + logK + Kt;
      float logf2    = logA + 2 * z + (float)std::log(gsl_cdf_gaussian_P((double)-w * (t * z + 1), 1.0)) + logK + Kt;
      float p_over_q = (float)std::exp(logf1) + (float)std::exp(logf2);
      float ratio    = 1.0 / (1.0 + p_over_q);

      float u, X;
      // Main sampling loop; page 130 of the Windle PhD thesis
      while (1) {
         // Step 1: Sample X ? g(x|z)
         //u = R::runif(0.0,1.0);

         u = rand_unif_thread(r_thread);
         if (u < ratio) {
            // truncated exponential
            X = t + exprnd_thread(r_thread, 1.0) / K;
         }else{
            // truncated Inverse Gaussian
            X = tinvgauss_thread(r_thread, z, t);
         }
         // Step 2: Iteratively calculate Sn(X|z), starting at S1(X|z), until U ? Sn(X|z) for an odd n or U > Sn(X|z) for an even n
         int   i  = 1;
         float Sn = aterm(0, X, t);
         //double U = R::runif(0.0,1.0) * Sn;
         float U    = rand_unif_thread(r_thread) * Sn;
         int   asgn = -1;
         bool  even = false;
         while (1) {
            Sn = Sn + asgn * aterm(i, X, t);
            // Accept if n is odd
            if (!even && (U <= Sn)) {
               X = X * 0.25;
               return(X);
            }
            // Return to step 1 if n is even
            if (even && (U > Sn)) {
               break;
            }
            even = !even;
            asgn = -asgn;
            i++;
         }
      }
      return(X);
   }

   Eigen::MatrixXf pgdraw_thread(gsl_rng *r_thread, const Eigen::MatrixXf& b, const Eigen::MatrixXf& c) {
      int m = b.rows();
      int n = c.rows();

      //NumericVector y(n);
      Eigen::MatrixXf y = 0.0 * c;
      // Setup
      int k, j, bi = 1;
      if (m == 1) {
         bi = b(0, 0);
      }
      // Sample
      for (k = 0; k < n; k++) {
         if (m > 1) {
            bi = b(k, 0);
         }
         // Sample
         y(k, 0) = 0;
         for (j = 0; j < (int)bi; j++) {
            y(k, 0) += samplepg_thread(r_thread, c(k, 0));
         }
      }
      return(y);
   }

   float gsl_pnorm(float x, int log_flag) {
      float out = 0.0;

      if (log_flag != 1) {
         out = (float)gsl_cdf_gaussian_P((double)x, 1.0);
      }else{
         out = (float)std::log(gsl_cdf_gaussian_P((double)x, 1.0));
      }
      return(out);
   }

   void make_Eigen_doc2eta_lambda_vec() {
      doc2eta_table    = Eigen::MatrixXf::Ones(num_docs, num_topics);
      doc2lambda_table = Eigen::MatrixXf::Ones(num_docs, num_topics);
      //std::cout << doc2eta_table << std::endl;
      for (auto itr = doc2place_vec.begin(); itr != doc2place_vec.end(); ++itr) {
         int             doc      = itr->first;
         Eigen::MatrixXf temp_eta = doc2eta_vec[doc];
         //std::cout << doc << " " << temp_eta.transpose() << std::endl;
         //std::cout << doc2eta_table.block(doc, 0, 1, num_topics) << std::endl;
         Eigen::MatrixXf temp_lambda = doc2lambda_vec[doc];
         doc2eta_table.block(doc, 0, 1, num_topics)    = temp_eta.transpose();
         doc2lambda_table.block(doc, 0, 1, num_topics) = temp_lambda.transpose();
      }
   }

   void cgtm_decide_initial_variable() {
      for (auto itr = doc2place_vec.begin(); itr != doc2place_vec.end(); ++itr) {
         int doc = itr->first;
         doc2eta_vec[doc]    = initial_eta_vec;
         doc2lambda_vec[doc] = initial_lambda_vec;
      }
      mu_for_topicprop_prior    = Eigen::MatrixXf::Constant(num_topics, 1, 0.01);
      Sigma_for_topicprop_prior = Eigen::MatrixXf::Identity(num_topics, num_topics);
      Psi_for_topicprop_prior   = Eigen::MatrixXf::Identity(num_topics, num_topics);
      kappa_for_topicprop_prior = 0.1;
      mu_c    = Eigen::MatrixXf::Constant(num_topics, 1, 0.01);
      Sigma_c = Eigen::MatrixXf::Identity(num_topics, num_topics);
   }

   void cgtm_cal_sampling_parameter_for_lambda(int& doc, Eigen::MatrixXf& zeta, Eigen::MatrixXf& rho, float& Nd, const int verbose) {
      float temp = 0.0;

      for (int j = 0; j < num_topics; ++j) {
         temp += std::exp(doc2eta_vec[doc](j, 0));
      }
      for (int k = 0; k < num_topics; ++k) {
         zeta(k, 0) = std::log(temp - std::exp(doc2eta_vec[doc](k, 0)));
      }
      if (verbose == 1) {
         std::cout << "zeta " << zeta << std::endl;
      }
      rho = doc2eta_vec[doc] - zeta;
      if (verbose == 1) {
         std::cout << "rho " << rho << std::endl;
      }
      Nd = float(doc2place_vec[doc].size());
      if (verbose == 1) {
         std::cout << "Nd " << Nd << std::endl;
      }
   }

   Eigen::MatrixXf cgtm_cal_inverse_Sigma_c(Eigen::MatrixXf& Sigma_c, const int verbose) {
      Eigen::MatrixXf temp = Sigma_c.inverse();
      return(temp);
   }

   void cgtm_cal_sampling_parameter_for_eta(const int& doc, const int& topic, const Eigen::MatrixXf& sampling_lambda, Eigen::MatrixXf& zeta, float& cond_mu, float& cond_sigma, float& C_D_K, float& kappa_in_sampling, float& tau, float& gamma_in_sampling, Eigen::MatrixXf inv_Sigma_c, const int verbose) {
      cond_sigma = 1.0 / inv_Sigma_c(topic, topic);

      Eigen::MatrixXf partial_inv_Sigma_c = Eigen::MatrixXf::Zero(1, num_topics - 1);
      Eigen::MatrixXf partial_eta         = Eigen::MatrixXf::Zero(num_topics - 1, 1);
      Eigen::MatrixXf partial_mu_c        = Eigen::MatrixXf::Zero(num_topics - 1, 1);

      for (int j = 0; j < num_topics; ++j) {
         if (j < topic) {
            partial_inv_Sigma_c(0, j) = inv_Sigma_c(topic, j);
            partial_eta(j, 0)         = doc2eta_vec[doc](j, 0);
            partial_mu_c(j, 0)        = mu_c(j, 0);
         }else if (j > topic) {
            partial_inv_Sigma_c(0, j - 1) = inv_Sigma_c(topic, j);
            partial_eta(j - 1, 0)         = doc2eta_vec[doc](j, 0);
            partial_mu_c(j - 1, 0)        = mu_c(j, 0);
         }
      }

      Eigen::MatrixXf temp_mat = cond_sigma * partial_inv_Sigma_c * (partial_eta - partial_mu_c);
      cond_mu = mu_c(topic, 0) - float(temp_mat(0, 0));

      if (verbose == 1) {
         std::cout << "doc topic " << doc << " " << topic << std::endl;
         std::cout << "cond_sigma " << cond_sigma << std::endl;
         std::cout << "partial_inv_Sigma_c " << partial_inv_Sigma_c << std::endl;
         //std::cout << "swap_eta " << swap_eta << std::endl;
         //std::cout << "swap_mu_c " << swap_mu_c << std::endl;
         std::cout << "partial_eta " << partial_eta << std::endl;
         std::cout << "partial_mu_c " << partial_mu_c << std::endl;
         std::cout << "temp_mat (this must be scaler.) " << temp_mat << std::endl;
         std::cout << "cond_mu " << cond_mu << std::endl;
      }

      //C_D_K = 0.0;
      C_D_K = doc2topic_vec[doc](topic, 0);
      std::vector <int> place_vec = doc2place_vec[doc];

      kappa_in_sampling = C_D_K - float(place_vec.size()) / 2.0;
      float temp = 1.0 / cond_sigma;
      temp += sampling_lambda(topic, 0);//ここが更新したあとでないといけない？
      tau   = 1.0 / temp;

      if (verbose == 1) {
         std::cout << "C_D_K " << C_D_K << std::endl;
         std::cout << "kappa " << kappa_in_sampling << std::endl;
      }

      float temp2 = cond_mu / cond_sigma;
      temp2 += kappa_in_sampling;
      temp2 += sampling_lambda(topic, 0) * zeta(topic, 0);
      if (verbose == 1) {
         std::cout << "for equation 11 " << std::endl;
         std::cout << sampling_lambda(topic, 0) << std::endl;
         std::cout << zeta(topic, 0) << std::endl;
         std::cout << zeta << std::endl;
         std::cout << temp2 << std::endl;
         std::cout << "for equation 11 end" << std::endl;
      }
      gamma_in_sampling = temp2 * tau;
      if (verbose == 1) {
         std::cout << "gamma " << gamma_in_sampling << " tau " << tau << std::endl;
      }
   }

   void cgtm_update_mu_c_Sigma_c(int verbose) {
      Eigen::MatrixXf eta_mean = Eigen::MatrixXf::Zero(num_topics, 1);

      for (auto itr = doc2eta_vec.begin(); itr != doc2eta_vec.end(); ++itr) {
         Eigen::MatrixXf doc_eta = itr->second;
         eta_mean += doc_eta;
      }
      eta_mean = eta_mean / float(num_docs);
      if (verbose == 1) {
         std::cout << " eta_mean " << eta_mean << std::endl;
      }

      Eigen::MatrixXf temp3 = kappa_for_topicprop_prior * mu_for_topicprop_prior;
      temp3 += float(num_docs) * eta_mean;

      mu_c = temp3 / (kappa_for_topicprop_prior + float(num_docs));

      Eigen::MatrixXf temp4 = Psi_for_topicprop_prior;
      Eigen::MatrixXf temp5 = Eigen::MatrixXf::Zero(num_topics, num_topics);
      for (auto itr = doc2eta_vec.begin(); itr != doc2eta_vec.end(); ++itr) {
         Eigen::MatrixXf doc_eta = itr->second;
         temp5 += (doc_eta - eta_mean) * (doc_eta - eta_mean).transpose();
      }
      temp5 = temp5 / float(num_docs);
      if (verbose == 1) {
         std::cout << " temp5 " << temp5 << std::endl;
      }


      Eigen::MatrixXf temp6 = (eta_mean - mu_for_topicprop_prior) * (eta_mean - mu_for_topicprop_prior).transpose();
      temp6 = temp6 * kappa_for_topicprop_prior * float(num_docs) / (kappa_for_topicprop_prior + float(num_docs));

      if (verbose == 1) {
         std::cout << " temp6 " << temp6 << std::endl;
      }

      Sigma_c = temp4 + temp5 + temp6;
      if (verbose == 1) {
         std::cout << "mu_c " << mu_c << std::endl;
         std::cout << "Sigma_c " << Sigma_c << std::endl;
      }
   }

   void cgtm_collapsed_gibbs_sample_parallel_for_all(const int num_iteration, const int parallel_loop, int num_threads, const int verbose, const int verbose2) {
      if (verbose == 1) {
         std::cout << "Enter cgtm_collapsed_gibbs_sample_parallel_for_logistic_normal_parameters" << std::endl;
      }

      stop_hantei = 0;
      if (num_threads < 1) {
         num_threads = omp_get_max_threads();
         std::cout << "Using " << num_threads << " threads" << std::endl;
      }

      omp_set_num_threads(num_threads);

      const gsl_rng_type *T;
      gsl_rng **          r;
      gsl_rng_env_setup();
      T = gsl_rng_mt19937;
      r = (gsl_rng **)malloc(num_threads * sizeof(gsl_rng *));
      std::random_device rand_device;
      std::mt19937       mt(rand_device());
      for (int i = 0; i < num_threads; i++) {
         r[i] = gsl_rng_alloc(T);
         gsl_rng_set(r[i], mt());
      }
      if (verbose == 1) {
         std::cout << "After gsl_rng_set" << std::endl;
      }


      for (int itr_outer = 0; itr_outer < num_iteration; ++itr_outer) {
         if (stop_hantei == 1) {
            std::cout << "ERROR: COULD NOT SAMPLE TOPIC" << std::endl;
            break;
         }

         Eigen::MatrixXi doc_word_topic_update = Eigen::MatrixXi::Zero(parallel_loop, 3);
         std::unordered_map <int, std::vector <Eigen::MatrixXf> > lambda_eta_update;
         std::unordered_map <int, std::vector <int> >             topic_assignment_update, topic_assignment_update_correspond;
         if (verbose == 1) {
            std::cout << "before parallel" << std::endl;
         }

         std::unordered_map <int, float> topic2second_nume_denom, topic2third_nume;
         #pragma omp parallel for
         for (int i = 0; i < num_topics; ++i) {
            float second_nume_denom_t = glda_posterior_predictive_second_nume_denom(i, nu, kappa, topic2num, topic2mu, topic2Sigma, 0);
            float third_nume_t        = glda_posterior_predictive_third_nume(i, nu, kappa, topic2num, topic2mu, topic2Sigma, 0);
            #pragma omp critical
            {
               topic2second_nume_denom[i] = second_nume_denom_t;
               topic2third_nume[i]        = third_nume_t;
            }
         }

         #pragma omp parallel for
         for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
            if (stop_hantei != 1) {
               int thread_id  = omp_get_thread_num();
               int doc_thread = int(floor(float(num_docs) * gsl_rng_uniform(r[thread_id])));

               Eigen::MatrixXf zeta  = Eigen::MatrixXf::Zero(num_topics, 1);
               Eigen::MatrixXf rho   = Eigen::MatrixXf::Zero(num_topics, 1);
               float           Nd    = 0.0;
               clock_t         start = clock();

               if (1 == 1) {
                  std::vector <float> ratio;
                  if (verbose == 1) {
                     std::cout << "before cal sampling parameter for lambda" << std::endl;
                  }

                  cgtm_cal_sampling_parameter_for_lambda(doc_thread, zeta, rho, Nd, verbose);

                  clock_t after_cal_for_lambda = clock();

                  if (verbose == 1) {
                     std::cout << "after cal sampling parameter for lambda " << std::endl;
                  }
                  if (verbose2 == 1) {
                     std::cout << "time for cal sampling parameters for lambda" << double(after_cal_for_lambda - start) / CLOCKS_PER_SEC << std::endl;
                  }

                  Eigen::MatrixXf Nd_for_pgdraw;
                  Nd_for_pgdraw = Eigen::MatrixXf::Constant(num_topics, 1, 1.0) * Nd;
                  Eigen::MatrixXf sampling_lambda = Eigen::MatrixXf::Zero(num_topics, 1);

                  if (verbose == 1) {
                     std::cout << "before sampling lambda" << std::endl;
                  }

                  // KOKO GA CHIGAU
                  gsl_rng *r_thread = r[thread_id];
                  int      success  = 0;
                  for (int k = 0; k < max_try; ++k) {
                     sampling_lambda = pgdraw_thread(r_thread, Nd_for_pgdraw, rho);
                     int hantei = 0;
                     for (int l = 0; l < num_topics; ++l) {
                        if (sampling_lambda(l, 0) == float(1.0) / float(0.0) || sampling_lambda(l, 0) == float(-1.0) / float(0.0)) {
                           hantei = 1;
                        }
                     }
                     if (hantei == 0) {
                        success = 1;
                        break;
                     }
                  }
                  if (success == 0) {
                     #pragma omp critical
                     {
                        stop_hantei = 1;
                        std::cout << "ERROR PGDRAW INIFINTE, max_try is " << max_try << std::endl;
                     }
                  }
                  // END KOKO GA CHIGAU //

                  clock_t after_sampling_lambda = clock();

                  if (verbose == 1) {
                     std::cout << "after sampling lambda " << sampling_lambda << std::endl;
                  }
                  if (verbose2 == 1) {
                     std::cout << "time for sampling lambda " << double(after_sampling_lambda - after_cal_for_lambda) / CLOCKS_PER_SEC << std::endl;
                  }
                  if (verbose == 1) {
                     std::cout << "doc num " << doc_thread << std::endl;
                  }

                  if (sampling_lambda == Eigen::MatrixXf::Zero(num_topics, 1)) {
                     #pragma omp critical
                     {
                        stop_hantei = 1;
                        std::cout << "ERROR STOP LAMBDA" << std::endl;
                     }
                  }         // if(new_path < 0){

                  //std::cout << "End Step A" << std::endl;
                  // Step B: sample doc2eta_vec
                  Eigen::MatrixXf sampling_eta = Eigen::MatrixXf::Zero(num_topics, 1);

                  if (stop_hantei != 1) {
                     float cond_mu;
                     float cond_sigma;
                     float C_D_K;
                     float kappa_in_sampling;
                     float tau;
                     float gamma_in_sampling;

                     std::vector <int> topic_thread;
                     for (int k = 0; k < num_topics; ++k) {
                        topic_thread.push_back(k);
                     }
                     std::random_device seed_gen;
                     std::mt19937       engine(seed_gen());
                     std::shuffle(topic_thread.begin(), topic_thread.end(), engine);

                     clock_t ready_for_eta = clock();

                     if (verbose2 == 1) {
                        std::cout << "time for ready for eta " << double(ready_for_eta - after_sampling_lambda) / CLOCKS_PER_SEC << std::endl;
                     }

                     Eigen::MatrixXf inv_Sigma_c = cgtm_cal_inverse_Sigma_c(Sigma_c, verbose);

                     clock_t cal_inv_Sigma_c = clock();

                     if (verbose2 == 1) {
                        std::cout << "time for cal inv_Sigma_c " << double(cal_inv_Sigma_c - ready_for_eta) / CLOCKS_PER_SEC << std::endl;
                     }

                     for (int k = 0; k < num_topics; ++k) {
                        cgtm_cal_sampling_parameter_for_eta(doc_thread, topic_thread[k], sampling_lambda, zeta, cond_mu, cond_sigma, C_D_K, kappa_in_sampling, tau, gamma_in_sampling, inv_Sigma_c, verbose);
                        clock_t cal_parameter_for_eta = clock();
                        if (verbose2 == 1) {
                           std::cout << "time for cal parameter for eta " << double(cal_parameter_for_eta - cal_inv_Sigma_c) / CLOCKS_PER_SEC << std::endl;
                        }
                        double temp    = gsl_ran_gaussian(r[thread_id], double(tau)) + double(gamma_in_sampling);
                        float  eta_D_K = float(temp);

                        if (verbose == 1) {
                           std::cout << "==============" << std::endl;
                           std::cout << "tau for sampling " << tau << std::endl;
                           std::cout << "gamma for sampling " << gamma_in_sampling << std::endl;
                           std::cout << "eta_D_K " << eta_D_K << std::endl;
                           std::cout << "zeta " << zeta << std::endl;
                        }

                        clock_t cal_zeta = clock();

                        /*
                         * for (int j = 0; j < num_topics; ++j) {
                         * if (j != topic_thread[k]) {
                         *    zeta(j, 0)  = std::exp(zeta(j, 0));
                         *    zeta(j, 0) += std::exp(eta_D_K) - std::exp(zeta(topic_thread[k], 0));
                         *    std::cout << zeta(j, 0) << std::endl;
                         *    zeta(j, 0)  = std::log(zeta(j, 0));
                         * }
                         * }
                         */
                        float temp2 = 0.0;

                        for (int j = 0; j < num_topics; ++j) {
                           if (j != topic_thread[k]) {
                              temp2 += std::exp(doc2eta_vec[doc_thread](j, 0));
                           }else{
                              temp2 += std::exp(eta_D_K);
                           }
                        }
                        for (int i = 0; i < num_topics; ++i) {
                           if (i != topic_thread[k]) {
                              zeta(i, 0) = std::log(temp2 - std::exp(doc2eta_vec[doc_thread](i, 0)));
                           }else{
                              zeta(i, 0) = std::log(temp2 - std::exp(eta_D_K));
                           }
                        }
                        if (verbose == 1) {
                           std::cout << "==============" << std::endl;
                           std::cout << "zeta " << zeta << std::endl;
                        }
                        cal_inv_Sigma_c = clock();
                        if (verbose2 == 1) {
                           std::cout << "time for cal parameter for zeta after eta sampling " << double(cal_inv_Sigma_c - cal_zeta) / CLOCKS_PER_SEC << std::endl;
                        }
                        sampling_eta(topic_thread[k], 0) = eta_D_K;
                     }

                     if (verbose == 1) {
                        std::cout << " sampling eta " << sampling_eta << std::endl;
                     }

                     if (sampling_eta == Eigen::MatrixXf::Zero(num_topics, 1)) {
                        #pragma omp critical
                        {
                           stop_hantei = 1;
                           std::cout << "ERROR STOP ETA" << std::endl;
                        }
                     }        // if(new_path < 0){

                     std::vector <Eigen::MatrixXf> result;
                     result.push_back(sampling_lambda);
                     result.push_back(sampling_eta);

                     // Step C : update topic assignment.

                     std::vector <int> place_vec_thread = doc2place_vec[doc_thread];

                     // Create place_vec_thread shuffled
                     std::shuffle(place_vec_thread.begin(), place_vec_thread.end(), engine);
                     //int inner_count = 0;

                     std::vector <int> result2, result3;
                     result3 = place_vec_thread;


                     // for each word position
                     for (int itr_place = 0; itr_place < int(place_vec_thread.size()); ++itr_place) {
                        int             place = place_vec_thread[itr_place];
                        int             word  = doc_word_topic(place, 1);
                        Eigen::MatrixXf x     = embedding_matrix.row(word);

                        // Create super topic ratio
                        std::vector <float> topic_ratio;
                        std::vector <float> ratio;

                        //create sub topic ratio
                        for (int i = 0; i < num_topics; ++i) {
                           float temp = alpha_topic_vec(i) + doc2topic_vec[doc_thread](i, 0);
                           topic_ratio.push_back(temp);
                        }

                        // create ratio
                        float max_val = -9999999999999;
                        std::unordered_map <int, float> multi_t_prob;
                        for (int i = 0; i < num_topics; i++) {
                           multi_t_prob[i] = glda_posterior_predictive_fast(place, i, nu, kappa, topic2num, topic2mu, topic2Sigma, topic2second_nume_denom, topic2third_nume, 1, 0);
                        }

                        for (int i = 0; i < num_topics; ++i) {
                           float temp = std::log(topic_ratio[i]) + multi_t_prob[i];
                           ratio.push_back(temp);
                           if (max_val < temp) {
                              max_val = temp;
                           }
                        }

                        float agg_topic = 0;
                        for (int i = 0; i < int(ratio.size()); ++i) {
                           ratio[i]   = ratio[i] - max_val;
                           ratio[i]   = std::exp(ratio[i]);
                           agg_topic += ratio[i];
                        }

                        // sample
                        double prob[ratio.size()];                                                                         // Probability array
                        for (int i = 0; i < int(ratio.size()); ++i) {
                           prob[i] = double(ratio[i] / agg_topic);
                        }
                        unsigned int mult_op[ratio.size()];
                        int          num_sample = 1;
                        gsl_ran_multinomial(r[thread_id], ratio.size(), num_sample, prob, mult_op);
                        int new_topic = -1;
                        for (int i = 0; i < int(ratio.size()); ++i) {
                           if (mult_op[i] == 1) {
                              new_topic = i;
                              break;
                           }
                        }

                        result2.push_back(new_topic);
                     }


                     #pragma omp critical
                     {
                        //int hantei     = 1;
                        //int change_doc = -1;

                        doc_word_topic_update(itr_inner, 0)           = doc_thread;
                        lambda_eta_update[itr_inner]                  = result;
                        topic_assignment_update[itr_inner]            = result2;
                        topic_assignment_update_correspond[itr_inner] = result3;
                     }
                  }  //if(stop_hantei == 1)
               }     //if(1==1){
            }        //if(stop_hantei == 1)
         }           //for(int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner){


         // Update eta, lambda
         for (int itr_inner = 0; itr_inner < parallel_loop; ++itr_inner) {
            int doc = doc_word_topic_update(itr_inner, 0);

            Eigen::MatrixXf   sampling_lambda      = lambda_eta_update[itr_inner][0];
            Eigen::MatrixXf   sampling_eta         = lambda_eta_update[itr_inner][1];
            std::vector <int> topic_assignment_vec = topic_assignment_update[itr_inner];


            doc2lambda_vec[doc] = sampling_lambda;
            doc2eta_vec[doc]    = sampling_eta;
            std::vector <int> place_vec_thread = topic_assignment_update_correspond[itr_inner];
            for (int itr_place = 0; itr_place < int(place_vec_thread.size()); ++itr_place) {
               int place     = place_vec_thread[itr_place];
               int word      = doc_word_topic(place, 1);
               int topic     = doc_word_topic(place, 2);
               int new_topic = topic_assignment_vec[itr_place];

               // word embedding : x (1,embedding_dimension)
               Eigen::MatrixXf x = embedding_matrix.row(word);

               if (doc2topic_vec[doc](topic, 0) > 0.0) {
                  // Subtract tables
                  doc2topic_vec[doc](topic, 0) = doc2topic_vec[doc](topic, 0) - 1.0;

                  // Update tables
                  doc2topic_vec[doc](new_topic, 0) = doc2topic_vec[doc](new_topic, 0) + 1.0;

                  // Subtract parameters
                  glda_update_parameters_subtract(topic, word);

                  // Update parameters
                  glda_update_parameters_add(new_topic, word);

                  // Update assignments
                  doc_word_topic(place, 2) = new_topic;
               }
            }
         }
         // Update mu_c, Simga_c
         cgtm_update_mu_c_Sigma_c(verbose);
      }             //for(int itr = 0; itr < num_iteration; ++itr){

      if (verbose == 1) {
         std::cout << "Before Free memory" << std::endl;
      }
      // free memory
      for (int i = 0; i < num_threads; i++) {
         gsl_rng_free(r[i]);
      }
      if (verbose == 1) {
         std::cout << "After Free memory" << std::endl;
      }
   }     //cgtm_collapsed_gibbs_sample_parallel_for_all

   void cgtm_decide_initial_variable_test() {
      for (auto itr = doc2place_vec_test.begin(); itr != doc2place_vec_test.end(); ++itr) {
         int doc = itr->first;
         doc2eta_vec_test[doc]    = initial_eta_vec;
         doc2lambda_vec_test[doc] = initial_lambda_vec;
      }
      //mu_for_topicprop_prior    = Eigen::MatrixXf::Constant(num_topics, 1, 0.01);
      //Sigma_for_topicprop_prior = Eigen::MatrixXf::Identity(num_topics, num_topics);
      //Psi_for_topicprop_prior   = Eigen::MatrixXf::Identity(num_topics, num_topics);
      //kappa_for_topicprop_prior = 0.1;
      //mu_c    = Eigen::MatrixXf::Constant(num_topics, 1, 0.01);
      //Sigma_c = Eigen::MatrixXf::Identity(num_topics, num_topics);
   }

   void cgtm_cal_sampling_parameter_for_lambda_test(Eigen::MatrixXf& sampling_eta, int& doc, Eigen::MatrixXf& zeta, Eigen::MatrixXf& rho, float& Nd, const int verbose) {
      float temp = 0.0;
      for (int j = 0; j < num_topics; ++j) {
         temp += std::exp(sampling_eta(j, 0));
      }
      for (int k = 0; k < num_topics; ++k) {
         zeta(k, 0) = std::log(temp - std::exp(sampling_eta(k, 0)));
      }
      rho = sampling_eta - zeta;
      Nd = float(doc2place_vec_test[doc].size());
   }

   void cgtm_cal_sampling_parameter_for_eta_test(Eigen::MatrixXf& sampling_eta, Eigen::MatrixXf& topic_vec_sub,const int& doc, const int& topic, const Eigen::MatrixXf& sampling_lambda, Eigen::MatrixXf& zeta, float& cond_mu, float& cond_sigma, float& C_D_K, float& kappa_in_sampling, float& tau, float& gamma_in_sampling, Eigen::MatrixXf inv_Sigma_c, const int verbose) {

      cond_sigma = 1.0 / inv_Sigma_c(topic, topic);
      Eigen::MatrixXf partial_inv_Sigma_c = Eigen::MatrixXf::Zero(1, num_topics - 1);
      Eigen::MatrixXf partial_eta         = Eigen::MatrixXf::Zero(num_topics - 1, 1);
      Eigen::MatrixXf partial_mu_c        = Eigen::MatrixXf::Zero(num_topics - 1, 1);

      for (int j = 0; j < num_topics; ++j) {
         if (j < topic) {
            partial_inv_Sigma_c(0, j) = inv_Sigma_c(topic, j);
            partial_eta(j, 0)         = sampling_eta(j, 0);
            partial_mu_c(j, 0)        = mu_c(j, 0);
         }else if (j > topic) {
            partial_inv_Sigma_c(0, j - 1) = inv_Sigma_c(topic, j);
            partial_eta(j - 1, 0)         = sampling_eta(j, 0);
            partial_mu_c(j - 1, 0)        = mu_c(j, 0);
         }
      }

      Eigen::MatrixXf temp_mat = cond_sigma * partial_inv_Sigma_c * (partial_eta - partial_mu_c);
      cond_mu = mu_c(topic, 0) - float(temp_mat(0, 0));

      //C_D_K = doc2topic_vec[doc](topic, 0);
      C_D_K = topic_vec_sub(topic, 0);
      std::vector <int> place_vec = doc2place_vec_test[doc];

      kappa_in_sampling = C_D_K - float(place_vec.size()) / 2.0;
      float temp = 1.0 / cond_sigma;
      temp += sampling_lambda(topic, 0);//ここが更新したあとでないといけない？
      tau   = 1.0 / temp;
      float temp2 = cond_mu / cond_sigma;
      temp2 += kappa_in_sampling;
      temp2 += sampling_lambda(topic, 0) * zeta(topic, 0);
      gamma_in_sampling = temp2 * tau;
   }

   void cgtm_evaluate_held_out_log_likelihood(const std::vector <int> doc_test_ids, const int num_iteration, const int burn_in, const float path_thres, const int approx, int num_threads, int verbose) {

      if (num_threads < 1) {
         num_threads = omp_get_max_threads();
         std::cout << "Using " << num_threads << " threads" << std::endl;
      }

      omp_set_num_threads(num_threads);
      int max_num_threads = omp_get_max_threads();
      const gsl_rng_type *T;
      gsl_rng **          r;
      gsl_rng_env_setup();
      T = gsl_rng_mt19937;
      r = (gsl_rng **)malloc(max_num_threads * sizeof(gsl_rng *));
      std::random_device rand_device;
      std::mt19937       mt(rand_device());
      for (int i = 0; i < max_num_threads; i++) {
         r[i] = gsl_rng_alloc(T);
         gsl_rng_set(r[i], mt());
      }
      create_topic2word_map();

      float counter = 0.0;
      #pragma omp parallel for
      for (int d = 0; d < int(doc_test_ids.size()); ++d) {
         int thread_id = omp_get_thread_num();
         gsl_rng *r_thread = r[thread_id];
         std::unordered_map <int, std::unordered_map <int, float> > topic2word_map_thread = topic2word_map;
         std::unordered_map <int, int>             update_topic_thread;
         int             doc_test = doc_test_ids[d];
         int             doc_size = int(doc2place_vec_test[doc_test].size());
         Eigen::MatrixXi doc_word_topic_thread = Eigen::MatrixXi::Zero(doc_size, 8);
         for (int i = 0; i < doc_size; ++i) {
            int place_t = doc2place_vec_test[doc_test][i];
            doc_word_topic_thread(i, 0) = doc_word_topic_test(place_t, 0);
            doc_word_topic_thread(i, 1) = doc_word_topic_test(place_t, 1);
            doc_word_topic_thread(i, 2) = doc_word_topic_test(place_t, 2);
         }

         // for every word position
         Eigen::MatrixXf heldout_prob = Eigen::MatrixXf::Zero(doc_word_topic_thread.rows(), 1);
         std::unordered_map <int, std::unordered_map <int, float> > place2topic_weight_thread;
         for (int up_to = 0; up_to < int(doc_word_topic_thread.rows()); ++up_to) {
            std::unordered_map <int, float>           topic2num_thread   = topic2num;
            std::unordered_map <int, Eigen::MatrixXf> topic2mu_thread    = topic2mu;
            std::unordered_map <int, Eigen::MatrixXf> topic2Sigma_thread = topic2Sigma;
            Eigen::MatrixXf topic_vec_sub = Eigen::MatrixXf::Zero(num_topics, 1);
            Eigen::MatrixXi doc_word_topic_thread_temp = Eigen::MatrixXi::Zero(up_to + 1, 10);

            for (int i = 0; i < (up_to + 1); ++i) {
               doc_word_topic_thread_temp(i, 0) = doc_word_topic_thread(i, 0);
               doc_word_topic_thread_temp(i, 1) = doc_word_topic_thread(i, 1);
               doc_word_topic_thread_temp(i, 2) = doc_word_topic_thread(i, 2);
               int topic = doc_word_topic_thread_temp(i, 2);
               if (i == up_to) {
                  topic_vec_sub(topic, 0) += 1.0;
               }else{
                  for (auto itr3 = place2topic_weight_thread[i].begin(); itr3 != place2topic_weight_thread[i].end(); ++itr3) {
                     topic = itr3->first;
                     float val = itr3->second;
                     topic_vec_sub(topic, 0) += val;
                  }
               } //for (auto itr3 = place2topic_weight_thread[i].begin()
            }    //for (int i = 0; i < (up_to + 1); ++i) {

            float agg   = 0;
            float count = 0;
            update_topic_thread.clear();
            Eigen::MatrixXf sampling_lambda = Eigen::MatrixXf::Zero(num_topics, 1);
            Eigen::MatrixXf sampling_eta = Eigen::MatrixXf::Zero(num_topics, 1);
            std::unordered_map <int, float> topic_weight_thread;
            for (int rep = 0; rep < num_iteration; ++rep) {
                Eigen::MatrixXf zeta  = Eigen::MatrixXf::Zero(num_topics, 1);
                Eigen::MatrixXf rho   = Eigen::MatrixXf::Zero(num_topics, 1);
                float           Nd    = float(doc_size);
                std::vector <float> ratio;
                cgtm_cal_sampling_parameter_for_lambda_test(sampling_eta, doc_test, zeta, rho, Nd, verbose);
                Eigen::MatrixXf Nd_for_pgdraw;
                Nd_for_pgdraw = Eigen::MatrixXf::Constant(num_topics, 1, 1.0) * Nd;
                for (int k = 0; k < max_try; ++k) {
                    sampling_lambda = pgdraw_thread(r_thread, Nd_for_pgdraw, rho);
                    int hantei = 0;
                    for (int l = 0; l < num_topics; ++l) {
                        if (sampling_lambda(l, 0) == float(1.0) / float(0.0) || sampling_lambda(l, 0) == float(-1.0) / float(0.0)) {
                            hantei = 1;
                         }
                    }
                    if (hantei == 0) {
                        break;
                    }
                }

                // Step B: sample doc2eta_vec
                float cond_mu;
                float cond_sigma;
                float C_D_K;
                float kappa_in_sampling;
                float tau;
                float gamma_in_sampling;

                std::vector <int> topic_thread;
                for (int k = 0; k < num_topics; ++k) {
                    topic_thread.push_back(k);
                }
                std::random_device seed_gen;
                std::mt19937       engine(seed_gen());
                std::shuffle(topic_thread.begin(), topic_thread.end(), engine);
                Eigen::MatrixXf inv_Sigma_c = cgtm_cal_inverse_Sigma_c(Sigma_c, verbose);

                for (int k = 0; k < num_topics; ++k) {
                    cgtm_cal_sampling_parameter_for_eta_test(sampling_eta,topic_vec_sub,doc_test, topic_thread[k], sampling_lambda, zeta, cond_mu, cond_sigma, C_D_K, kappa_in_sampling, tau, gamma_in_sampling, inv_Sigma_c, verbose);
                    double temp    = gsl_ran_gaussian(r[thread_id], double(tau)) + double(gamma_in_sampling);
                    float  eta_D_K = float(temp);
                    float temp2 = 0.0;
                    for (int j = 0; j < num_topics; ++j) {
                        if (j != topic_thread[k]) {
                            temp2 += std::exp(doc2eta_vec_test[doc_test](j, 0));
                        }else{
                            temp2 += std::exp(eta_D_K);
                        }
                    }
                    for (int i = 0; i < num_topics; ++i) {
                        if (i != topic_thread[k]) {
                            zeta(i, 0) = std::log(temp2 - std::exp(doc2eta_vec_test[doc_test](i, 0)));
                        }else{
                            zeta(i, 0) = std::log(temp2 - std::exp(eta_D_K));
                        }
                    }
                    sampling_eta(topic_thread[k], 0) = eta_D_K;
                }

                std::unordered_map <int, float> topic_weight_thread;
                int             place     = int(doc_word_topic_thread_temp.rows()) - 1;
                int             word      = doc_word_topic_thread_temp(place, 1);
                int             old_topic = doc_word_topic_thread_temp(place, 2);
                Eigen::MatrixXf x         = embedding_matrix.row(word);

                // Create super topic ratio
                std::vector <float> topic_ratio;
                ratio.clear();

                //create sub topic ratio
                for (int i = 0; i < num_topics; ++i) {
                    //float temp = alpha_topic_vec(i) + doc2topic_vec[doc_thread](i, 0);
                    float temp = alpha_topic_vec(i) + topic_vec_sub(i, 0);
                    topic_ratio.push_back(temp);
                }

                // create ratio
                float max_val = -9999999999999;
                std::unordered_map <int, float> multi_t_prob;
                for (int i = 0; i < num_topics; ++i) {
                    multi_t_prob[i] = glda_posterior_predictive_test(doc_word_topic_thread_temp, place, i, nu, kappa, topic2num_thread, topic2mu_thread, topic2Sigma_thread, 1, 0);
                    float temp = std::log(topic_ratio[i]) + multi_t_prob[i];
                    ratio.push_back(temp);
                    if (max_val < temp) {
                        max_val = temp;
                    }
                }
                float agg_topic = 0;
                for (int i = 0; i < int(ratio.size()); ++i) {
                    ratio[i]   = ratio[i] - max_val;
                    ratio[i]   = std::exp(ratio[i]);
                    agg_topic += ratio[i];
                }
                double prob[ratio.size()];
                for (int i = 0; i < int(ratio.size()); ++i) {
                    prob[i] = double(ratio[i] / agg_topic);
                }
                unsigned int mult_op[ratio.size()];
                int          num_sample = 1;
                gsl_ran_multinomial(r[thread_id], ratio.size(), num_sample, prob, mult_op);
                int new_topic = -1;
                for (int i = 0; i < int(ratio.size()); ++i) {
                    if (mult_op[i] == 1) {
                        new_topic = i;
                        break;
                    }
                }
                topic_vec_sub(old_topic, 0)         -= 1.0;
                topic_vec_sub(new_topic, 0)         += 1.0;
                doc_word_topic_thread_temp(place, 2) = new_topic;
                doc_word_topic_thread(place, 2)      = new_topic;
                if (rep > burn_in) {
                   for (int j = 0; j < int(ratio.size()); ++j) {
                       new_topic = j;
                       if(topic2word_map_thread[new_topic].find(word)!=topic2word_map_thread[new_topic].end()){
                         agg += float(prob[j]) * (topic2word_map_thread[new_topic][word] + beta) / (voc * beta + topic2num_thread[new_topic]);
                       }else{
                         agg += float(prob[j]) * beta / (voc * beta + topic2num_thread[new_topic]);
                       }
                   }
                   count += 1.0;
                   if (topic_weight_thread.find(new_topic) == topic_weight_thread.end()) {
                      topic_weight_thread[new_topic] = 1.0;
                   }else{
                      topic_weight_thread[new_topic] += 1.0;
                   }
                } //if (rep > burn_in) {
             }    //for(int rep = 0;rep < num_iteration;++rep){

            float prob_2 = agg / count;
            heldout_prob(up_to, 0) = prob_2;

            // Update place2topic_weight_thread
            float agg2 = 0.0;
            for (auto itr3 = topic_weight_thread.begin(); itr3 != topic_weight_thread.end(); ++itr3) {
               agg2 += 1.0;
            }
            for (auto itr3 = topic_weight_thread.begin(); itr3 != topic_weight_thread.end(); ++itr3) {
               itr3->second = itr3->second / agg2;
            }
            place2topic_weight_thread[up_to] = topic_weight_thread;
         }//for (int up_to = 0; up_to < int(doc_word_topic_thread.rows()); ++up_to) {
         #pragma omp critical
         {
            doc_test2heldout_prob[doc_test] = heldout_prob;
            counter += 1;
            std::cout << "Finished: " << counter / float(doc_test_ids.size()) << std::endl;
         } //#pragma omp critical
      }    //for (int k = 0; k < int(doc_test_ids.size()); ++k)
   }

   // CGTM End//

   // AUXILIARY FUNCTIONS TO CHECK THE USE OF MULTICORE EIGEN
   static std::string GetClassName() {
      return("GHLDA");
   }

   void SetName(const std::string& input) {
      name = input;
   }

   std::string GetName() {
      return(name);
   }
};

// THE OLD WAY
PYBIND11_PLUGIN(ghlda) {
   py::module m("ghlda", "");
   // THE NEW WAY TO WRITE IT
   //PYBIND11_MODULE(gaussian_hlda,m){

   //  Binders
   py::bind_map <std::unordered_map <int, Eigen::MatrixXf> >(m, "IntMatrixMap");
   py::bind_map <std::unordered_map <long long, Eigen::MatrixXf> >(m, "LongMatrixMap");
   py::bind_map <std::unordered_map <int, std::unordered_map <int, float> > >(m, "IntIntMap");

   // Class GHLDA
   py::class_ <GHLDA>(m, "GHLDA", py::dynamic_attr())
   .def(py::init <const std::string&>())

   // functions
   .def("prob_mass_mvt", &GHLDA::prob_mass_mvt)
   .def("calc_variance", &GHLDA::calc_variance)

   // variables
   .def_readwrite("doc_word_topic", &GHLDA::doc_word_topic)
   .def_readwrite("embedding_matrix", &GHLDA::embedding_matrix)
   .def_readwrite("embedding_dimension", &GHLDA::embedding_dimension)
   .def_readwrite("embedding_center", &GHLDA::embedding_center)
   .def_readwrite("num_path", &GHLDA::num_path)
   .def_readwrite("num_topics", &GHLDA::num_topics)
   .def_readwrite("num_depth", &GHLDA::num_depth)
   .def_readwrite("num_docs", &GHLDA::num_docs)
   .def_readwrite("topic2num", &GHLDA::topic2num)
   .def_readwrite("topic2mu", &GHLDA::topic2mu)
   .def_readwrite("topic2Sigma", &GHLDA::topic2Sigma)
   .def_readwrite("kappa", &GHLDA::kappa)
   .def_readwrite("nu", &GHLDA::nu)
   .def_readwrite("Psi", &GHLDA::Psi)
   .def_readwrite("alpha_topic_vec", &GHLDA::alpha_topic_vec)
   .def_readwrite("min_diagonal_val", &GHLDA::min_diagonal_val)
   .def_readwrite("doc2topic_vec", &GHLDA::doc2topic_vec)
   .def_readwrite("stop_hantei", &GHLDA::stop_hantei)
   .def_readwrite("voc", &GHLDA::voc)
   .def_readwrite("beta", &GHLDA::beta)
   .def_readwrite("num_0", &GHLDA::num_0)
   .def_readwrite("mu_0", &GHLDA::mu_0)
   .def_readwrite("Sigma_0", &GHLDA::Sigma_0)

   // LDA
   .def("lda_calc_tables_parameters_from_assign", &GHLDA::lda_calc_tables_parameters_from_assign)
   .def("lda_collapsed_gibbs_sample_parallel", &GHLDA::lda_collapsed_gibbs_sample_parallel)
   .def("lda_evaluate_held_out_log_likelihood", &GHLDA::lda_evaluate_held_out_log_likelihood)

   // HLDA
   .def_readwrite("level2eta_in_HLDA", &GHLDA::level2eta_in_HLDA)
   .def_readwrite("topic2word_map", &GHLDA::topic2word_map)
   .def_readwrite("zero_topic2word_map", &GHLDA::zero_topic2word_map)
   .def_readwrite("word2topic_vec", &GHLDA::word2topic_vec)
   .def_readwrite("word2level_vec", &GHLDA::word2level_vec)
   .def("hlda_calc_tables_parameters_from_assign", &GHLDA::hlda_calc_tables_parameters_from_assign)
   .def("hlda_ncrp_calc_tables_parameters_from_assign", &GHLDA::hlda_ncrp_calc_tables_parameters_from_assign)
   .def("hlda_fixed_collapsed_gibbs_sample_parallel", &GHLDA::hlda_fixed_collapsed_gibbs_sample_parallel)
   .def("hlda_ncrp_collapsed_gibbs_sample_parallel", &GHLDA::hlda_ncrp_collapsed_gibbs_sample_parallel)
   .def("check_topic2word_map", &GHLDA::check_topic2word_map)
   .def("hlda_ncrp_evaluate_held_out_log_likelihood",&GHLDA::hlda_ncrp_evaluate_held_out_log_likelihood)

   // GLDA
   .def_readwrite("id2word", &GHLDA::id2word)
   .def_readwrite("zero_mu", &GHLDA::zero_mu)
   .def_readwrite("zero_Sigma", &GHLDA::zero_Sigma)
   .def_readwrite("doc_word_topic_test", &GHLDA::doc_word_topic_test)
   .def_readwrite("topic2word_prob", &GHLDA::topic2word_prob)
   .def_readwrite("topic2word_prob_debug", &GHLDA::topic2word_prob_debug)
   .def_readwrite("doc_test2heldout_prob", &GHLDA::doc_test2heldout_prob)
   .def("glda_calc_tables_parameters_from_assign", &GHLDA::glda_calc_tables_parameters_from_assign)
   .def("glda_collapsed_gibbs_sample_parallel", &GHLDA::glda_collapsed_gibbs_sample_parallel)
   .def("glda_collapsed_gibbs_sample_parallel_fast", &GHLDA::glda_collapsed_gibbs_sample_parallel_fast)
   .def("glda_collapsed_gibbs_sample_parallel_faster", &GHLDA::glda_collapsed_gibbs_sample_parallel_faster)
   .def("glda_evaluate_held_out_log_likelihood", &GHLDA::glda_evaluate_held_out_log_likelihood)
   .def("create_topic2word_prob", &GHLDA::create_topic2word_prob)
   .def("create_topic2word_map",&GHLDA::create_topic2word_map)

   // GHLDA-fixed
   .def("path_level2topic", &GHLDA::path_level2topic)
   .def_readwrite("path_dict", &GHLDA::path_dict)
   .def_readwrite("path2num", &GHLDA::path2num)
   .def_readwrite("alpha_level_vec", &GHLDA::alpha_level_vec)
   .def_readwrite("doc2level_vec", &GHLDA::doc2level_vec)
   .def_readwrite("doc2place_vec", &GHLDA::doc2place_vec)
   .def_readwrite("doc2place_vec_test", &GHLDA::doc2place_vec_test)
   .def_readwrite("sample_counter", &GHLDA::sample_counter)
   .def_readwrite("path_gamma", &GHLDA::path_gamma)
   .def_readwrite("level_gamma", &GHLDA::level_gamma)
   .def_readwrite("level2Psi", &GHLDA::level2Psi)
   .def_readwrite("word2cdf", &GHLDA::word2cdf)
   .def_readwrite("mix_ratio", &GHLDA::mix_ratio)
   .def_readwrite("level_allocation_type", &GHLDA::level_allocation_type)
   .def_readwrite("hyper_m", &GHLDA::hyper_m)
   .def_readwrite("hyper_pi", &GHLDA::hyper_pi)
   .def("create_initial_hierarchy", &GHLDA::create_initial_hierarchy)
   .def("create_initial_hierarchy_rev", &GHLDA::create_initial_hierarchy_rev)
   .def("create_initial_assignment", &GHLDA::create_initial_assignment)
   .def("create_doc2place_vec", &GHLDA::create_doc2place_vec)
   .def("create_doc2place_vec_test", &GHLDA::create_doc2place_vec_test)
   .def("ghlda_posterior_predictive_robust", &GHLDA::ghlda_posterior_predictive_robust)
   .def("ghlda_posterior_predictive_single_robust", &GHLDA::ghlda_posterior_predictive_single_robust)
   .def("ghlda_calc_tables_parameters_from_assign", &GHLDA::ghlda_calc_tables_parameters_from_assign)
   .def("ghlda_collapsed_gibbs_sample_parallel", &GHLDA::ghlda_collapsed_gibbs_sample_parallel)
   .def("ghlda_collapsed_gibbs_sample_parallel_fast", &GHLDA::ghlda_collapsed_gibbs_sample_parallel_fast)

   // GHLDA-ncrp
   .def_readwrite("max_depth_allowed", &GHLDA::max_depth_allowed)
   .def_readwrite("new_path_dict_debug", &GHLDA::new_path_dict_debug)
   .def_readwrite("extended_path_dict_debug", &GHLDA::extended_path_dict_debug)
   .def_readwrite("doc2path", &GHLDA::doc2path)
   .def_readwrite("topic2level", &GHLDA::topic2level)
   .def_readwrite("gamma", &GHLDA::gamma)
   .def_readwrite("doc_word_topic_test", &GHLDA::doc_word_topic_test)
   .def_readwrite("rescale", &GHLDA::rescale)
   .def_readwrite("inner_rescale", &GHLDA::inner_rescale)
   .def_readwrite("topic_dict", &GHLDA::topic_dict)
   .def("ghlda_ncrp_calc_tables_parameters_from_assign", &GHLDA::ghlda_ncrp_calc_tables_parameters_from_assign)
   .def("ghlda_ncrp_collapsed_gibbs_sample_parallel", &GHLDA::ghlda_ncrp_collapsed_gibbs_sample_parallel)
   .def("ghlda_ncrp_evaluate_held_out_log_likelihood",&GHLDA::ghlda_ncrp_evaluate_held_out_log_likelihood)

   // CGTM
   .def_readwrite("doc2lambda_vec", &GHLDA::doc2lambda_vec)
   .def_readwrite("doc2eta_vec", &GHLDA::doc2eta_vec)
   .def_readwrite("doc2eta_table", &GHLDA::doc2eta_table)
   .def_readwrite("doc2lambda_table", &GHLDA::doc2lambda_table)
   .def_readwrite("Sigma_c", &GHLDA::Sigma_c)
   .def_readwrite("mu_c", &GHLDA::mu_c)
   .def_readwrite("initial_eta_vec", &GHLDA::initial_eta_vec)
   .def_readwrite("initial_lambda_vec", &GHLDA::initial_lambda_vec)
   .def_readwrite("max_try", &GHLDA::max_try)
   .def("cgtm_decide_initial_variable", &GHLDA::cgtm_decide_initial_variable)
   .def("pgdraw_thread", &GHLDA::pgdraw_thread)
   .def("cgtm_update_mu_c_Sigma_c", &GHLDA::cgtm_update_mu_c_Sigma_c)
   .def("cgtm_cal_sampling_parameter_for_lambda", &GHLDA::cgtm_cal_sampling_parameter_for_lambda)
   .def("cgtm_cal_sampling_parameter_for_eta", &GHLDA::cgtm_cal_sampling_parameter_for_eta)
   .def("make_Eigen_doc2eta_lambda_vec", &GHLDA::make_Eigen_doc2eta_lambda_vec)
   .def("cgtm_collapsed_gibbs_sample_parallel_for_all", &GHLDA::cgtm_collapsed_gibbs_sample_parallel_for_all)
   .def("cgtm_decide_initial_variable_test",&GHLDA::cgtm_decide_initial_variable_test)
   .def("cgtm_evaluate_held_out_log_likelihood",&GHLDA::cgtm_evaluate_held_out_log_likelihood)

   // SOME AUXILIARY FUNCTIONS
   .def_static("GetClassName", &GHLDA::GetClassName)
   .def("SetName", &GHLDA::SetName)
   .def("GetName", &GHLDA::GetName);

   // THE OLD WAY // COMMENT OUT WHEN USING NEW WAY
   return(m.ptr());
}

int main() {
   return(0);
}
