include("manager.jl")

usage() = println("Must supply a command. One of --run, --setup, --clean")

pwd = splitpath(abspath("."))[end]
if pwd != "KH_3"
    println("Must run from `KH_3` directory.")
    exit(1)
end

if length(ARGS) == 0
    usage()
    exit(1)
end
command = ARGS[1]

if command == "--run"
    id = parse(Int64, ARGS[2])
    CampaignManager.run_sim(; id)
elseif command == "--setup"
    CampaignManager.setup_campaign()
elseif command == "--clean"
    CampaignManager.cleanup()
else
    usage()
    exit(1)
end


