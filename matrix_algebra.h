#include<iostream>
#include<cstdio>
#include <cstdlib>
#include <vector> 
#include <chrono>	//for time measurement
#include <cmath>	//for calculating power & NaN
#include <omp.h>	//for parallel
#include <stdio.h>
#include <numeric>    // for average
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator

using namespace std;

    // Some shortcuts
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Coo<ValueType, IndexType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;

// Figure out where to run the code
const auto gpu = gko::CudaExecutor::create(0, gko::OmpExecutor::create(), true);


template <typename T>
T vector_norm(vector<T> a){
	T value = 0.0;
	for(int i = 0; i < a.size(); ++i) {
		value += a[i] * a[i];
	}
	value = sqrt(value);
	return value;
}

template <typename T>
T vector_norm(vector<vector<T>> A){
	T value = 0.0;
	for(int i = 0; i < A.size(); ++i) {
		for(int j = 0; j < A.size(); ++j) {
			value += A[i][j] * A[i][j];
		}
	}
	value = sqrt(value);
	return value;
}

template <typename T>
vector<vector<T>> transpose(vector<vector<T>> &A) {
		int rows = A.size();
		int column = A[0].size();
		vector< vector<T> > result(column, vector<T>(rows,0));
		int i, j;
		for (i = 0; i < column; ++i) {
			for (j = 0; j < rows; ++j){
				result[i][j]=A[j][i];
			}
		}
		return result;
}

template <typename T>
vector<vector<T>> transpose(vector<T> &a) {
		int array_size = a.size();
		vector< vector<T> > result(1, vector<T>(array_size,0.0));
		for (int i = 0; i < array_size; ++i) {
			result[0][i]=a[i];
		}
		return result;
}

//a dot b
template <typename T>
T dot_product(vector<T> a, vector<T> b){
	T result = 0.0;
	for(int i = 0; i < a.size(); ++i) {
		result += a[i] * b[i];
	}
	return result;
}

//A dot b = mxn nx1 
template <typename T>
vector<T> dot_product(vector <vector<T>> A, vector<T> b){
	vector<T> result(A.size(), 0.0);
	
	for(int i = 0; i < A.size(); ++i) {
		T value = 0.0;
		for(int j = 0; j < A[0].size(); ++j) {
			value += A[i][j] * b[j];
		}
		result[i] = value;
	}
	return result;
}

//a dot B = nx1 1xm 
template <typename T>
vector <vector<T>> dot_product(vector<T> a, vector <vector<T>> B){
	int row_size = a.size();
	int col_size = B[0].size();
	vector <vector<T>> result(row_size, vector<T>(col_size, 0.0));
	
	for(int i = 0; i < row_size; ++i) {
		for(int j = 0; j < col_size; ++j) {
			result[i][j] =  a[i] * B[0][j];
		}
	}
	return result;
}

//A dot B
template <typename T>
vector<vector<T>> dot_product(vector<vector<T>> &A, vector<vector<T>> &B) {
		int A_rows= A.size();
		int B_col= B[0].size();
		int A_col= A[0].size();
		vector< vector<T> > result(A_rows, vector<T>(B_col,0));
		int i,j;
		for (i = 0; i < A_rows; ++i) {
			for (j = 0; j < B_col; ++j){
				T s = 0;
				for (int k = 0; k <A_col; k++) {
					s += A[i][ k] * B[k][j];
				}
				result[i][j]=s;
			}
		}
		return result;
	}

template <typename T>
void printMatrix(vector<vector<T>> &myArray) {
	for(size_t x = 0;x < myArray.size(); ++x){
        for(size_t y = 0;y < myArray[x].size();++y){
            printf("  %f", myArray[x][y]);
			printf("  ");
        }
        cout << endl;
    }		
}

template <typename T>
void printMatrix(vector<T> &myArray) {
	for(size_t i = 0; i < myArray.size(); ++i){
        printf("  %f  \n", myArray[i]);
    }
}

void printMatrix(gko::matrix::Coo<ValueType> *A) {
	//Copy to CPU
	auto A2 = mtx::create(gpu->get_master());
	A2->copy_from(lend(A));
	
	auto nnz_total= A2->get_num_stored_elements();
    auto r = A2->get_const_row_idxs();
	auto c = A2->get_const_col_idxs();
	auto v = A2->get_const_values();
	int Ni = A2->get_size()[0];
	int Nj = A2->get_size()[1];
	
	for (int i = 0; i < Ni; i++) {
		for (int j = 0; j < Nj; j++) {
			int found = 0;
			for (int k = 0; k < nnz_total; k++) {
				if((i== r[k])&&(j == c[k])){
					std::cout<<v[k]<<" ";
					found = 1;
				}
			}
			if(found==0){
				std::cout<<"0 ";
			}
		}
		std::cout<<std::endl;
    }
}

//For 2D and 1D dense
void printMatrix(gko::matrix::Dense<ValueType> *A) {
	//Copy to CPU
	auto A2 = gko::matrix::Dense<ValueType>::create(gpu->get_master());
	A2->copy_from(lend(A));
	
	int Ni = A2->get_size()[0];
	int Nj = A2->get_size()[1];
	
	for (int i = 0; i < Ni; i++) {
		for (int j = 0; j < Nj; j++) {
			cout<<A2->at(i,j)<<" ";
		}
		std::cout<<std::endl;
    }
}

//Pending :matrix data should be symmetric
//Error using template
vector <vector<double>> read_matfile(string filename){
	std::ifstream fin(filename); //need to delete 31 top rows
	
	if (!fin){
		printf("Failed to read data!");
		vector <vector<double>> temp;
		return temp;
	}
	
	std::string line;
	int vidx = 0;
	
	std::vector<std::array<double, 4>> A_vec;
	vector <vector<double>> A;
	
	while (getline(fin, line)){
		std::istringstream sin(line);
		A_vec.push_back(std::array<double, 4>()); // last col to store "\n"
		vidx = A_vec.size() - 1;
		size_t i = 0; // Reset on each loop.
		while (sin >> A_vec[vidx][i++]){ 
		}
	}
	//Debugging
		//std::cout << "Size A_vec = "<< A_vec.size()<< std::endl;
		//for (int i = (start_data-1); i < (start_data+11); i++) {
		//	std::cout <<  A_vec[i][0]<< " "<<A_vec[i][1]<<" "<< A_vec[i][2]<< std::endl;
		//}
		
	//-----------------------------------------------
	//From Ginkgo (mtx)
	if(filename.substr(filename.find_last_of(".") + 1) == "mtx") {
		int start_data = 31 ;// first array that contain data
		int matrix_size = A_vec[(start_data-1)][0];	
		
		vector <vector<double>> A_output(matrix_size, vector<double>(matrix_size, 0.0));
		
		for (int i = start_data; i < A_vec.size(); i++) {
				int row = A_vec[i][0]-1; //array starts from 1
				int col = A_vec[i][1]-1; //array starts from 1
				A_output[row][col] = A_vec[i][2];
		}
		A= A_output; // copy data to outer scope
	} // end of if (file extension)
	
	//-----------------------------------------------
	//For Petsc (txt)
	if(filename.substr(filename.find_last_of(".") + 1) == "txt") {
		int start_data = 6 ;// first array location that contain data
		int end_data = 2 ;// number of last arrays that are not used
	
		cout << "Please provide matrix size = ";
		int matrix_size;
		cin >> matrix_size;
		cout <<"\n\n";
		vector <vector<double>> A_output(matrix_size, vector<double>(matrix_size, 0.0));
		
		for (int i = start_data; i < (A_vec.size()-end_data); i++) {
				int row = A_vec[i][0]-1; //array starts from 1
				int col = A_vec[i][1]-1; //array starts from 1
				A_output[row][col] = A_vec[i][2];
		}
		A= A_output; // copy data to outer scope
	} // end of if (file extension)
	//-----------------------------------------------
	return A;
}

//Error using template
vector<double> read_vecfile(string filename){
	std::ifstream fin(filename); //need to delete 2 top rows
	
	if (!fin){
		printf("Failed to read data!");
		vector<double> temp;
		return temp;
	}
	
	std::string line;
	std::vector<std::array<double, 2>> x0_vec;
	int vidx = 0;
	vector<double> x;
	while (getline(fin, line))	{
		std::istringstream sin(line);
		x0_vec.push_back(std::array<double, 2>()); // last col to store "\n"
		vidx = x0_vec.size() - 1;
		size_t i = 0; // Reset on each loop.
		while (sin >> x0_vec[vidx][i++]) { 
		}
	}
	//Debugging
		//std::cout << "Size x0_vec = "<< x0_vec.size()<< std::endl;
		//for (int i = (start_data-1); i < (start_data+8); i++) {
			//std::cout <<  x0_vec[i][0]<< " "<<x0_vec[i][1]<< std::endl;
		//}
	//-----------------------------------------------
	//From Ginkgo mtx
	if(filename.substr(filename.find_last_of(".") + 1) == "mtx") {
		int start_data = 2 ;// first array that contain data
		
		int matrix_size = x0_vec[1][0];
		vector <double> x0_output(matrix_size, 0.0);
		
		for (int i = start_data; i < x0_vec.size(); i++) {
				x0_output[i-start_data] = x0_vec[i][0];
		}
		x = x0_output;// copy data to outer scope
	} // end of if (file extension)
	//-----------------------------------------------
	
	//From Petsc (txt)
	if(filename.substr(filename.find_last_of(".") + 1) == "txt") {		
		int start_data = 2 ;// first array that contain data
		int matrix_size = x0_vec.size() - start_data;
		
		vector <double> x0_output(matrix_size, 0.0);
		
		for (int i = start_data; i < x0_vec.size(); i++) {
			x0_output[i-start_data] = x0_vec[i][0];
		}
		x = x0_output; // copy data to outer scope
	} // end of if (file extension)
	//-----------------------------------------------
	
	return x;
}

double sum_elements(gko::matrix::Dense<ValueType> *x){	
	//Copy to CPU
	//Todo : Do the computation in GPU instead of CPU
	auto x_master = vec::create(gpu->get_master());
	x_master->copy_from(x);
	double sum = 0.0;
	
	auto num_rows = x_master->get_size()[0];
    auto num_cols = x_master->get_size()[1];
    
    for (size_t row = 0; row < num_rows; ++row) {
        for (size_t col = 0; col < num_cols; ++col) {
            sum += x_master->at(row, col);
        }
    }
	
	return sum;
}

double sum_elements(gko::matrix::Coo<ValueType> *x){
	//Copy to CPU
	//Todo : Do the computation in GPU instead of CPU
	auto x_master = mtx::create(gpu->get_master());
	x_master->copy_from(x);
	auto v = x_master->get_const_values();
	double sum = 0.0;
	
	for (int i = 0; i < x->get_num_stored_elements(); ++i) {
		sum += v[i];
		
    }
	
	return sum;
}

double sum_elements(std::vector<double>& vec){
	double sum = 0.0;
	auto num_rows = vec.size();
    
    for (size_t row = 0; row < num_rows; ++row) {
         sum += vec[row];
    }

	return sum;
}

double sum_elements(std::vector<std::vector<double>>& vec){
	double sum = 0.0;
	auto num_rows = vec.size();
	auto num_cols = vec[0].size();
    
    for (size_t i = 0; i < num_rows; ++i) {
		for (size_t j = 0; j < num_cols; ++j) {
         sum += vec[i][j];
		}
	}
	return sum;
}


/*
int main(int argc, char *argv[]){	
	vector <vector<double>> A;
	A.push_back({1,3,4,5});
	A.push_back({2,6,7,10});
	A.push_back({3,8,9,12});
	std::cout<<"Matrix A = \n";
	printMatrix(A);
	
	vector <vector<double>> B;
	B.push_back({3,-3, 4, 7});
	B.push_back({4, -6,2,-1});
	B.push_back({5, -8, 3, 9});
	
	std::cout<<"Matrix B = \n";
	printMatrix(B);
	
	vector<double> a {1, -6, 9};
	vector<double> b {5, 6, 19};
	
	auto B_transpose = transpose(B);
	std::cout<<"B_transpose = \n";
	printMatrix(B_transpose);
	
	std::cout<<"a = \n";
	printMatrix(a);
	std::cout<<"a_transpose = \n";
	auto a_transpose = transpose(a);
	printMatrix(a_transpose);
	
	std::cout<<"|a| = "<< vector_norm(a) << endl;
	std::cout<<"|B| = "<< vector_norm(B) << endl;
	
	auto a_dot_b = dot_product(a,b);
	std::cout<<"a_dot_b = "<<a_dot_b<<endl;
	
	vector<double> c {1, -6, 9, -7};
	vector <vector<double>> C_mat;
	C_mat.push_back({3,-3, 4, 7});
	auto c_dot_C_mat = dot_product(c,C_mat);
	std::cout<<"c_dot_C_mat = "<<endl;
	printMatrix(c_dot_C_mat);
	
	vector<double> A_dot_c = dot_product(A,c);
	std::cout<<"A_dot_c = "<<endl;
	printMatrix(A_dot_c);
	
	auto A_dot_BT = dot_product(A,B_transpose);
	std::cout<<"A_dot_B^T = "<<endl;
	printMatrix(A_dot_BT);
	
	//Test for read_vecfile() and read_matfile()--------
	//Reading from Ginkgo mtx file
	auto A_mtx = read_matfile("data/A.mtx");
	//printMatrix(A_output);
	
	auto x0_final_mtx = read_vecfile("data/x0_final.mtx");
	//printMatrix(x0_final_mtx);
	
	// Checking multiplication A*x
	vector <double> A_dot_x_mtx = dot_product(A_mtx, x0_final_mtx);
	cout << "A_dot_x (mtx) = " <<endl;
	printMatrix(A_dot_x_mtx);
	
	//Reading from Petsc txt file
	auto A_txt = read_matfile("data/A_host.txt");
	//freopen("A_output.txt", "w", stdout);
	//printMatrix(A_output);
	//fclose (stdout);
	auto x0_txt = read_vecfile("data/solution_host.txt");
	//freopen("x_output.txt", "w", stdout);
	//printMatrix(x0_final_output);
	//fclose (stdout);
	// Checking multiplication A*x
	vector <double> A_dot_x_txt = dot_product(A_txt, x0_txt);
	cout << "A_dot_x (txt) = " <<endl;
	printMatrix(A_dot_x_txt);
	
	
}*/