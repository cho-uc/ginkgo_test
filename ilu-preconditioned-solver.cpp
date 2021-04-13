#include <ginkgo/ginkgo.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

    // Some shortcuts
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;

    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Coo<ValueType, IndexType>;
    using gmres = gko::solver::Gmres<ValueType>;
	
const auto gpu = gko::CudaExecutor::create(0, gko::OmpExecutor::create(), true);

std::shared_ptr<vec> run_ILU(std::shared_ptr<mtx> A, gko::matrix::Dense<ValueType> *b) {
	
	int row_size = A->get_size()[0];
	int col_size = A->get_size()[1];
	auto x = vec::create(gpu, gko::dim<2>(row_size, 1));
    // Generate incomplete factors using ParILU
    auto par_ilu_fact =
        gko::factorization::ParIlu<ValueType, IndexType>::build().on(gpu);
    // Generate concrete factorization for input matrix
   
	auto par_ilu = par_ilu_fact->generate(A);

    // Generate an ILU preconditioner factory by setting lower and upper
    // triangular solver - in this case the exact triangular solves
    auto ilu_pre_factory =
        gko::preconditioner::Ilu<gko::solver::LowerTrs<ValueType, IndexType>,
                                 gko::solver::UpperTrs<ValueType, IndexType>,
                                 false>::build()
            .on(gpu);

    // Use incomplete factors to generate ILU preconditioner
    auto ilu_preconditioner = ilu_pre_factory->generate(gko::share(par_ilu));

    // Use preconditioner inside GMRES solver factory
    // Generating a solver factory tied to a specific preconditioner makes sense
    // if there are several very similar systems to solve, and the same
    // solver+preconditioner combination is expected to be effective.
    const RealValueType reduction_factor{1e-7};
    auto ilu_gmres_factory =
        gmres::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1000u).on(gpu),
                gko::stop::ResidualNormReduction<ValueType>::build()
                    .with_reduction_factor(reduction_factor)
                    .on(gpu))
            .with_generated_preconditioner(gko::share(ilu_preconditioner))
            .on(gpu);
	//works until here
	
    // Generate preconditioned solver for a specific target system
    //auto ilu_gmres = ilu_gmres_factory->generate(A);
	
	auto ilu_gmres = ilu_gmres_factory->generate(A);

    // Solve system
    ilu_gmres->apply(gko::lend(b), gko::lend(x));
	
	return x;
}

int main(int argc, char *argv[]) {
    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    // Read data
	//Create A
	auto A = gko::share(mtx::create(gpu));
	int row_size = 3;
	int col_size = 3;
	gko::matrix_assembly_data<ValueType> data_mat(gko::dim<2>{row_size, col_size});
	
	data_mat.set_value(0, 0, 2);
	data_mat.set_value(0, 1, 2);
	data_mat.set_value(0, 2, 4);
	data_mat.set_value(1, 0, 1);
	data_mat.set_value(1, 1, 3);
	data_mat.set_value(1, 2, 5);
	data_mat.set_value(2, 0, 5);
	data_mat.set_value(2, 1, 2);
	data_mat.set_value(2, 2, 8);
	
	A->read(data_mat);
	
	std::cout<<"mat A = " <<std::endl;
	write(std::cout, lend(A));
	
	//Create b
	auto b = vec::create(gpu, gko::dim<2>(row_size, 1));
	gko::matrix_assembly_data<ValueType> data_vec(gko::dim<2>{row_size, 1});
	
	data_vec.set_value(0, 0, 1);
	data_vec.set_value(1, 0, 2);
	data_vec.set_value(2, 0, 3);
	
	b->read(data_vec);
	
	auto x = run_ILU((std::shared_ptr<mtx>)A, lend(b));
    // Print solution
    std::cout << "Solution (x):\n";
    write(std::cout, gko::lend(x));
	
	auto A_dot_x = gko::matrix::Dense<ValueType>::create(gpu, gko::dim<2>{3, 1});
        
	A->apply(gko::lend(x), gko::lend(A_dot_x));
	std::cout << "\t A_dot_x:\n";
    write(std::cout, gko::lend(A_dot_x));
	
}
