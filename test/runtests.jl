using LogitSamplers
using Test
using Random

@testset "LogitSamplers.jl" begin

    @testset "sample.jl" begin

        @testset "Basic operations" begin
            logits = randn(10)
            create_rng(seed=123) = Random.MersenneTwister(seed)

            @test logitsample(logits) isa Integer
            @test logitsample(create_rng(), logits) == only(logitsample_categorical(create_rng(), logits))
            @test logitsample(create_rng(), logits) == logitsample(create_rng(), logits)

            logits2 = [logits logits]
            @test logitsample(logits2) isa CartesianIndex{2}
            @test logitsample(logits2, dims=1) isa Matrix{CartesianIndex{2}}
            @test logitsample_categorical(create_rng(), logits2) != logitsample(create_rng(), logits2)
            @test logitsample_categorical(create_rng(), logits2) == logitsample_categorical(create_rng(), logits2, dims=1)
            @test logitsample_categorical(logits2, dims=1) isa Matrix{Int}
        end

        @testset "Statistical properties" begin
            n_samples = 10000
            logits = log.([0.25, 0.75])
            counts = zeros(Int, 2)
            for _ in 1:n_samples
                idx = only(logitsample_categorical(logits))
                counts[idx] += 1
            end
            
            probs = counts ./ n_samples
            @test isapprox(probs[2] / probs[1], 3.0, rtol=0.1)
        end

        @testset "Edge cases" begin
            large_logits = [1000.0, -1000.0]
            @test logitsample(large_logits) == 1

            equal_logits = zeros(3)
            samples = [logitsample(equal_logits) for _ in 1:1000]
            @test length(unique(samples)) == 3
        end

    end

    @testset "transforms.jl" begin

        count_remaining(logits) = count(!isinf, logits)

        @testset "Temperature" begin
            @test Temperature(1.0) isa Temperature

            t = Temperature(1.0)
            @test repr(MIME("text/plain"), t) == repr(t)

            logits = log.([0.1, 0.2, 0.3, 0.4])

            @test count_remaining(Temperature(0.5)(logits)) == 4
            @test count_remaining(Temperature(1.0)(logits)) == 4
            @test count_remaining(Temperature(2.0)(logits)) == 4
        end
        
        @testset "Top_pk" begin
            @test Top_pk(0.5, 3) isa Top_pk
            @test Top_p(0.5) isa Top_pk
            @test Top_k(3) isa Top_pk


            logits = log.([0.1, 0.2, 0.3, 0.4])

            @testset "Top_pk" begin
                @test count_remaining(Top_pk(0.30, 3)(logits)) == 1
                @test count_remaining(Top_pk(0.50, 1)(logits)) == 1
                @test count_remaining(Top_pk(0.50, 3)(logits)) == 2
                @test count_remaining(Top_pk(0.80, 2)(logits)) == 2
                @test count_remaining(Top_pk(0.80, 3)(logits)) == 3
                @test count_remaining(Top_pk(0.95, 3)(logits)) == 3
                @test count_remaining(Top_pk(0.95, 4)(logits)) == 4
                @test count_remaining(Top_pk(1.00, 4)(logits)) == 4

                @test_throws DomainError Top_pk(-1.0, 3)(logits)
                @test_throws DomainError Top_pk(2.0, 3)(logits)

                @test_throws BoundsError Top_pk(0.5, 5)(logits)
                @test_throws BoundsError Top_pk(0.5, 0)(logits)
            end

            @testset "Top_p" begin
                @test count_remaining(Top_p(0.30)(logits)) == 1
                @test count_remaining(Top_p(0.50)(logits)) == 2
                @test count_remaining(Top_p(0.80)(logits)) == 3
                @test count_remaining(Top_p(0.95)(logits)) == 4
                @test count_remaining(Top_p(1.00)(logits)) == 4

                @test_throws DomainError Top_p(-1.0)(logits)
                @test_throws DomainError Top_p(2.0)(logits)
            end

            @testset "Top_k" begin
                @test count_remaining(Top_k(1)(logits)) == 1
                @test count_remaining(Top_k(2)(logits)) == 2
                @test count_remaining(Top_k(3)(logits)) == 3
                @test count_remaining(Top_k(4)(logits)) == 4

                @test_throws BoundsError Top_k(5)(logits)
                @test_throws BoundsError Top_k(0)(logits)
            end

            @testset "batched" begin
                @test all(==(3) ∘ count_remaining, eachcol(Top_pk(0.80, 3)([logits logits .+ 10])))
            end
        end

        @testset "Min_p" begin
            @test Min_p(0.5) isa Min_p

            logits = log.([0.1, 0.2, 0.3, 0.4])

            @test count_remaining(Min_p(1.0)(logits)) == 1
            @test count_remaining(Min_p(0.8)(logits)) == 1
            @test count_remaining(Min_p(0.6)(logits)) == 2
            @test count_remaining(Min_p(0.4)(logits)) == 3
            @test count_remaining(Min_p(0.1)(logits)) == 4
            @test count_remaining(Min_p(0.0)(logits)) == 4
        end

        @testset "Top_nσ" begin
            @test Top_nσ(1.0) isa Top_nσ

            logits = log.([0.1, 0.2, 0.3, 0.4])

            @test count_remaining(Top_nσ(0.0)(logits)) == 1
            @test count_remaining(Top_nσ(1.0)(logits)) == 2
            @test count_remaining(Top_nσ(2.0)(logits)) == 3
            @test count_remaining(Top_nσ(3.0)(logits)) == 4
        end

    end

end
