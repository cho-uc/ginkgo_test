/*
Compare matrix multiplication b = A.x
between GPU and CPU
	GPU : Ginkgo apply(), new implementation matmul()
	CPU : std::vector

Conclusion : 
	- produce correct result if decimal digit is low but inexact when decimal digit is high
	- apply() is inexact but apply2() is exact
*/
#include <ginkgo/ginkgo.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <math.h>
#include <string>
#include <vector>
#include "matrix_algebra.h"


int main(int argc, char *argv[]){
	
	int m_size = 1000;
	auto A = mtx::create(gpu); //create coo mat
	auto x = vec::create(gpu); //create dense vec
	auto A_dot_x = vec::create(gpu); //create dense vec
	auto r_plus_A_dot_x = vec::create(gpu); //create dense vec
	
	gko::matrix_assembly_data<ValueType> data_A(gko::dim<2>{m_size, m_size});
	for (int i=0; i<m_size; i++) {
		for (int j=0; j<m_size; j++) {
			data_A.set_value(i, j, i*0.00235325+j*1.214324);
			//data_A.set_value(i, j, i*0.512+j*0.31);
		}
	}
    A->read(data_A);
	
	gko::matrix_assembly_data<ValueType> data_x(gko::dim<2>{m_size, 1});
	for (int i=0; i<m_size; i++) {
		data_x.set_value(i, 0, i*3.2777888);
		//data_x.set_value(i, 0, i+2.7);
	}
    x->read(data_x);
	A_dot_x->read(data_x);
	
	gko::matrix_assembly_data<ValueType> data_r(gko::dim<2>{m_size, 1});
	for (int i=0; i<m_size; i++) {
		data_r.set_value(i, 0, i*1.124242367);
		//data_r.set_value(i, 0, i+0.17);
	}
	
	r_plus_A_dot_x->read(data_r);
	
	lend(A)->apply(lend(x), lend(A_dot_x));	//Ginkgo original
	//lend(A)->matmul(lend(x), lend(A_dot_x));	//new implementation
	lend(A)->apply2(lend(x), lend(r_plus_A_dot_x));	//Ginkgo original
	//lend(A)->matmul2(lend(x), lend(r_plus_A_dot_x));	//new implementation
	
	cout<<"GPU version (Ginkgo)		A*x"<<endl;
	cout<<std::setprecision(30)<<"sum_elements A = " <<sum_elements(lend(A))<<endl;
	cout<<std::setprecision(30)<<"sum_elements x = " <<sum_elements(lend(x))<<endl;
	cout<<std::setprecision(30)<<"sum_elements A_dot_x = " <<sum_elements(lend(A_dot_x))<<endl;
	cout<<std::setprecision(30)<<"sum_elements r_plus_A_dot_x = " <<sum_elements(lend(r_plus_A_dot_x))<<endl;
	//printMatrix(lend(A_dot_x));
	
	//Compared to CPU calculation
	vector<double> x_vec (m_size);
	vector<double> r_plus_A_dot_x_vec (m_size);
	
	for (int i=0; i<m_size; i++) {
		x_vec[i] = i*3.2777888;
		//x_vec[i] = i+2.7;
	}
	
	for (int i=0; i<m_size; i++) {
		r_plus_A_dot_x_vec[i] = i*1.124242367;
		//r_plus_A_dot_x_vec[i] = i+0.17;
	}
	
	vector<vector<double>> A_vec (m_size);
	
	for (int i=0; i<m_size; i++) {
		A_vec[i] = vector<double>(m_size);
		
		for (int j=0; j<m_size; j++) {
			A_vec[i][j]= i*0.00235325+j*1.214324;
			//A_vec[i][j]= i*0.512+j*0.312;
		}
	}
	
	auto A_dot_x_vec = dot_product(A_vec, x_vec);
	
	for (int i=0; i<m_size; i++) {
		r_plus_A_dot_x_vec[i] += A_dot_x_vec[i];
	}
	
	cout<< "======================"<<endl;
	cout<<"CPU version (std::vector)	A*x"<<endl;
	cout<<std::setprecision(30)<<"sum_elements A = " <<sum_elements(A_vec)<<endl;
	cout<<std::setprecision(30)<<"sum_elements x = " <<sum_elements(x_vec)<<endl;
	cout<<std::setprecision(30)<<"sum_elements A_dot_x = " <<sum_elements(A_dot_x_vec)<<endl;
	cout<<std::setprecision(30)<<"sum_elements r_plus_A_dot_x = " <<sum_elements(r_plus_A_dot_x_vec)<<endl;
	//printMatrix(A_dot_x_vec);
}
