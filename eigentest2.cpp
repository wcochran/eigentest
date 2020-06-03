#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>
#include <vector>

int main(int argc, char *argv[]) {
	constexpr int K = 20;

	using Keypoints = std::vector<Eigen::Vector3f>;

	constexpr int N = 632;

	std::vector<Keypoints> A;
	std::vector<Keypoints> B;
	std::vector<Eigen::Matrix3f> Fun;
	std::vector<float> residuals;

	auto getval = []() -> float {
		static int i = 0;
		return std::cos(0.618*i++);
	};

	for (int i = 0; i < N; i++) {
		Keypoints U, V;
		for (int j = 0; j < K; j++) {
			U.emplace_back(Eigen::Vector3f(getval(), getval(), 1));
			V.emplace_back(Eigen::Vector3f(getval(), getval(), 1));
		}
		A.emplace_back(std::move(U));
		B.emplace_back(std::move(V));
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			Eigen::Matrix3f F;
			const float x = getval();
			const float y = getval();
			const float z = getval();
			F << 0, -z, y,  z, 0, -x,  -y, x, 0;
			F.normalize();
			Fun.emplace_back(std::move(F));
		}
	}
	
	constexpr int repeat = 1;

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	for (int I = 0; I < repeat; I++) {

		residuals.clear();

		for (int i1 = 0; i1 < N; i1++) {
			const Keypoints&  S = A[i1];
			Eigen::Matrix<float,3,Eigen::Dynamic> P(3,K);
			for (int i = 0; i < K; i++)
				for (int j = 0; j < 3; j++)
					P(j,i) = S[i](j);
			for (int i2 = 0; i2 < N; i2++) {
				const Keypoints&  T = B[i2];
				Eigen::Matrix<float,3,Eigen::Dynamic> Q(3,K);
				for (int i = 0; i < K; i++)
					for (int j = 0; j < 3; j++)
						Q(j,i) = T[i](j);
				const Eigen::Matrix3f& F = Fun[i1*N + i2];
				Eigen::Matrix<float,3,Eigen::Dynamic> L(3,K);
				L.noalias() = F*Q;

				const float residual2 = (L.transpose() * P).diagonal().array().square().mean();

				residuals.push_back(residual2);
			}
		}

	}

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
	std::cout << "duration: " << duration << " milliseconds" << std::endl;

	return 0;
}
