#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>

int main(int argc, char *argv[]) {
	constexpr int K = 20;
	Eigen::Matrix<float,3,Eigen::Dynamic> L(3,K);
	Eigen::Matrix<float,3,Eigen::Dynamic> P(3,K);

	for (int i = 0; i < K; i++) {
		L(0,i) = std::cos(0.618*i);
		L(1,i) = std::sin(0.618*i);
		L(2,i) = i;
		P(0,i) = 1.618*i;
		P(1,i) = 1.618*i*3;
		P(2,i) = 1;
	}

	constexpr int repeatCount = 100*632*631;

	float residual2;

	//
	// Case 1: Looping.
	//
	{
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < repeatCount; i++) {
			float distance = 0;
			for (int q = 0; q < K; q++){
				const Eigen::Vector3f& line = L.col(q);
				const Eigen::Vector3f& point = P.col(q);
				const float d = line.dot(point);
				distance += d*d;
			}
			residual2 = distance / K;
		}
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
		std::cout << "CASE 1: " << duration << " milliseconds" << std::endl;
		std::cout << "solution = " << residual2 << std::endl;
	}

	//
	// Case 2: Reduction 1
	//
	{
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < repeatCount; i++) {
			residual2 = (L.array() * P.array()).colwise().sum().square().mean();
		}
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
		std::cout << "CASE 2: " << duration << " milliseconds" << std::endl;
		std::cout << "solution = " << residual2 << std::endl;
	}

	//
	// Case 3: Reduction 2
	//
	{
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < repeatCount; i++) {
			residual2 = L.cwiseProduct(P).array().colwise().sum().array().square().mean();
		}
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
		std::cout << "CASE 3: " << duration << " milliseconds" << std::endl;
		std::cout << "solution = " << residual2 << std::endl;
	}

	//
	// Case 4: Reduction 3
	//
	{
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < repeatCount; i++) {
			residual2 = (L.transpose() * P).diagonal().array().square().mean();
		}
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
		std::cout << "CASE 4: " << duration << " milliseconds" << std::endl;
		std::cout << "solution = " << residual2 << std::endl;
	}

	return 0;
}
