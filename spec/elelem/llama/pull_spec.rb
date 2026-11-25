# frozen_string_literal: true

require "digest"
require "json"
require "stringio"
require_relative "../../../lib/elelem/llama/pull"
require_relative "local_origin"

RSpec.describe Elelem::Llama::Pull do
  weights = Random.new(1).bytes(300_000)
  commit = "c" * 40

  around do |example|
    Dir.mktmpdir { |dir| @dir = File.join(dir, "models") and example.run }
  ensure
    @origin&.close
  end

  let(:output) { StringIO.new }
  let(:files) { ["tiny-Q4_K_M.gguf", "tiny-Q8_0.gguf", "mmproj-Q4_K_M.gguf", "README.md"] }
  let(:sha256) { Digest::SHA256.hexdigest(weights) }
  let(:listing) do
    { sha: commit, siblings: files.map { { rfilename: it, size: weights.bytesize, lfs: { sha256: } } } }
  end

  def reply(...) = LocalOrigin.reply(...)

  def origin(&blob)
    @origin ||= LocalOrigin.new do |path, headers|
      path.start_with?("/api/") ? reply(body: JSON.generate(listing)) : (blob || whole).call(path, headers)
    end
  end

  def whole = ->(*) { reply(body: @served || Random.new(1).bytes(300_000)) }

  def ranged
    lambda do |_, headers|
      from = headers["range"].to_s[/\d+/].to_i
      body = Random.new(1).bytes(300_000).byteslice(from..)
      next reply(body:) if from.zero?

      reply("206 Partial Content", body:, headers: { "Content-Range" => "bytes #{from}-299999/300000" })
    end
  end

  def pull(*args, &)
    http = Elelem::Llama::Http.new(schemes: ["http"])
    hub = Elelem::Llama::Hub.new(http:, origin: origin(&).uri)
    described_class.new(args, dir: @dir, hub:, output:).run
  end

  def saved(name = "tiny-Q4_K_M.gguf") = File.join(@dir, name)

  def part = "#{saved}.#{sha256[0, 12]}.part"

  def put(path, content)
    FileUtils.mkdir_p(@dir)
    File.binwrite(path, content)
  end

  it "downloads the Q4_K_M model from the listed commit and makes it the default" do
    pull("me/tiny")

    expect(File.binread(saved)).to eq(weights)
    expect(File.identical?(saved("default.gguf"), saved)).to be(true)
    expect(origin.requests.last.first).to eq("/me/tiny/resolve/#{commit}/tiny-Q4_K_M.gguf")
    expect(Dir.children(@dir)).to contain_exactly("tiny-Q4_K_M.gguf", "default.gguf")
  end

  it "downloads the quant it is asked for" do
    pull("me/tiny:q8_0")

    expect(File.binread(saved("tiny-Q8_0.gguf"))).to eq(weights)
  end

  it "keeps the default it finds" do
    put(saved("mine.gguf"), "mine")
    File.symlink("mine.gguf", saved("default.gguf"))
    pull("me/tiny")

    expect(File.readlink(saved("default.gguf"))).to eq("mine.gguf")
    expect(output.string).to include("--default")
  end

  it "replaces the default when told to" do
    put(saved("mine.gguf"), "mine")
    File.symlink("mine.gguf", saved("default.gguf"))
    pull("me/tiny", "--default")

    expect(File.readlink(saved("default.gguf"))).to eq("tiny-Q4_K_M.gguf")
  end

  it "keeps a default that points at a model on an unmounted disk" do
    put(saved("other.gguf"), "other")
    File.symlink("/unmounted/mine.gguf", saved("default.gguf"))
    pull("me/tiny")

    expect(File.readlink(saved("default.gguf"))).to eq("/unmounted/mine.gguf")
  end

  it "never replaces a default that is also linked from a backup" do
    put(saved("default.gguf"), "weights")
    File.link(saved("default.gguf"), File.join(File.dirname(@dir), "backup.gguf"))

    expect { pull("me/tiny", "--default") }.to raise_error(/is a model and not a link/)
  end

  it "leaves nothing behind when the default is a directory" do
    FileUtils.mkdir_p(saved("default.gguf"))

    expect { pull("me/tiny", "--default") }.to raise_error(/is a model and not a link/)
    expect(Dir.children(@dir)).to contain_exactly("default.gguf", "tiny-Q4_K_M.gguf")
  end

  it "never replaces a default that is the model itself" do
    put(saved("default.gguf"), "weights")

    expect { pull("me/tiny", "--default") }.to raise_error(/is a model and not a link/)
    expect(File.read(saved("default.gguf"))).to eq("weights")
  end

  it "downloads nothing twice" do
    pull("me/tiny")
    pull("me/tiny")

    expect(origin.requests.map(&:first).grep(/resolve/).size).to eq(1)
  end

  it "never overwrites a different file of the same name" do
    put(saved, "other weights")

    expect { pull("me/tiny") }.to raise_error(/different file/)
    expect(File.read(saved)).to eq("other weights")
  end

  it "never overwrites a file of the same size with other weights" do
    put(saved, "x" * weights.bytesize)

    expect { pull("me/tiny") }.to raise_error(/different file/)
  end

  it "never overwrites a file that appears while it downloads" do
    expect do
      pull("me/tiny") do
        put(saved, "other weights")
        reply(body: weights)
      end
    end.to raise_error(/different file/)

    expect(File.read(saved)).to eq("other weights")
  end

  it "accepts the same file appearing while it downloads" do
    pull("me/tiny") do
      put(saved, weights)
      reply(body: weights)
    end

    expect(Dir.children(@dir)).to contain_exactly("tiny-Q4_K_M.gguf", "default.gguf")
  end

  it "never replaces a link to a model on an unmounted disk" do
    FileUtils.mkdir_p(@dir)
    File.symlink("/unmounted/tiny.gguf", saved)

    expect { pull("me/tiny") }.to raise_error(/different file/)
    expect(File.readlink(saved)).to eq("/unmounted/tiny.gguf")
  end

  it "refuses to resume from another offset than it asked for" do
    put(part, weights.byteslice(0, 100_000))
    range = { "Content-Range" => "bytes 5-299999/300000" }

    expect { pull("me/tiny") { reply("206 Partial Content", body: weights.byteslice(5..), headers: range) } }
      .to raise_error(/HTTP 206/)
  end

  it "forgets a stale part when the server ignores the range and then drops" do
    put(part, "s" * 100_000)
    blob = ranged
    dropped = false
    pull("me/tiny") do |path, headers|
      next blob.call(path, headers) if dropped

      dropped = true
      reply(body: weights.byteslice(0, 10), length: weights.bytesize)
    end

    expect(File.binread(saved)).to eq(weights)
  end

  it "starts over from a part larger than the model" do
    put(part, "s" * (weights.bytesize + 1))
    pull("me/tiny")

    expect(File.binread(saved)).to eq(weights)
  end

  it "asks again for the range after a redirect, and for no compression" do
    put(part, weights.byteslice(0, 100_000))
    blob = ranged
    pull("me/tiny") do |path, headers|
      path == "/moved" ? blob.call(path, headers) : reply("302 Found", headers: { "Location" => "/moved" })
    end

    expect(origin.requests.last.last).to include("range" => "bytes=100000-", "accept-encoding" => "identity")
  end

  it "gives up on a redirect loop" do
    expect { pull("me/tiny") { reply("302 Found", headers: { "Location" => "/again" }) } }
      .to raise_error(/too many redirects/)
  end

  it "allows only https by default" do
    expect { Elelem::Llama::Http.new.get(URI("http://127.0.0.1/")) { nil } }.to raise_error(/over http /)
  end

  it "reads the free space of a mount point with a space in its name" do
    report = <<~DF
      Filesystem 1024-blocks Used Available Capacity Mounted on
      /dev/disk4s1 1000 400 600 40% /Volumes/My Disk
    DF

    expect(Elelem::Llama::FreeSpace.available(report)).to eq(600 * 1024)
  end

  it "measures the free space of this machine" do
    expect(Elelem::Llama::FreeSpace.new.in(Dir.tmpdir)).to be_positive
  end

  it "resumes from what an earlier pull left" do
    put(part, weights.byteslice(0, 100_000))
    pull("me/tiny", &ranged)

    expect(File.binread(saved)).to eq(weights)
    expect(origin.requests.last.last["range"]).to eq("bytes=100000-")
  end

  it "starts over when the server ignores the range" do
    put(part, "stale" * 100)
    pull("me/tiny")

    expect(File.binread(saved)).to eq(weights)
  end

  it "asks for nothing when an earlier pull left a complete part" do
    put(part, weights)
    pull("me/tiny")

    expect(origin.requests.map(&:first).grep(/resolve/)).to be_empty
    expect(File.binread(saved)).to eq(weights)
  end

  it "continues by itself when the connection drops halfway" do
    dropped = false
    blob = ranged
    pull("me/tiny") do |path, headers|
      next blob.call(path, headers) if dropped

      dropped = true
      reply(body: weights.byteslice(0, 120_000), length: weights.bytesize)
    end

    expect(File.binread(saved)).to eq(weights)
  end

  it "keeps a short part when the server keeps hanging up" do
    expect { pull("me/tiny") { reply(body: weights.byteslice(0, 10), length: weights.bytesize) } }
      .to raise_error(/stopped early/)

    expect(File.size(part)).to be < weights.bytesize
    expect(File.exist?(saved)).to be(false)
  end

  it "deletes a download with the wrong checksum" do
    expect { pull("me/tiny") { reply(body: "x" * weights.bytesize) } }.to raise_error(/checksum mismatch/)

    expect(Dir.children(@dir)).to be_empty
  end

  it "stops a download larger than listed" do
    expect { pull("me/tiny") { reply(body: "#{weights}more") } }.to raise_error(/larger than expected/)
  end

  it "reports an error from the server" do
    expect { pull("me/tiny") { reply("503 Busy") } }.to raise_error(/HTTP 503/)
  end

  it "follows a relative redirect" do
    blob = whole
    pull("me/tiny") do |path, headers|
      path == "/moved" ? blob.call(path, headers) : reply("307 Temporary Redirect", headers: { "Location" => "/moved" })
    end

    expect(File.binread(saved)).to eq(weights)
  end

  it "refuses a redirect to another scheme" do
    expect { pull("me/tiny") { reply("302 Found", headers: { "Location" => "ftp://127.0.0.1/x" }) } }
      .to raise_error(/refusing to fetch over ftp/)
  end

  it "refuses to share a part with another process" do
    put(part, "")
    reader, writer = IO.pipe
    hold = "File.open(ARGV[0]) { it.flock(File::LOCK_EX); puts :held; $stdout.flush; sleep }"
    other = spawn(RbConfig.ruby, "-e", hold, part, out: writer)
    reader.gets

    expect { pull("me/tiny") }.to raise_error(/another process/)
  ensure
    Process.kill(:KILL, other)
    Process.wait(other)
  end

  it "needs room only for the missing part" do
    put(part, weights.byteslice(0, 250_000))
    hub = Elelem::Llama::Hub.new(http: Elelem::Llama::Http.new(schemes: ["http"]), origin: origin(&ranged).uri)
    space = Elelem::Llama::FreeSpace.new(host: Elelem::Llama::Host.new(system: "linux"), unix: ->(_) { 60_000 })
    model = hub.find(Elelem::Llama::ModelReference.parse("me/tiny"))

    expect(Elelem::Llama::Download.new(hub:, dir: @dir, free_space: space).call(model)).to eq(saved)
  end

  it "refuses when the disk is too full" do
    hub = Elelem::Llama::Hub.new(http: Elelem::Llama::Http.new(schemes: ["http"]), origin: origin.uri)
    space = Elelem::Llama::FreeSpace.new(host: Elelem::Llama::Host.new(system: "linux"), unix: ->(_) { 300_000 })
    model = hub.find(Elelem::Llama::ModelReference.parse("me/tiny"))

    expect { Elelem::Llama::Download.new(hub:, dir: @dir, free_space: space).call(model) }
      .to raise_error(/not enough disk space/)
  end

  it "names what a repository has when the quant is missing" do
    expect { pull("me/tiny:Q2_K") }.to raise_error(/has no Q2_K model; it has tiny-Q4_K_M.gguf, tiny-Q8_0.gguf/)
  end

  context "with a split model" do
    let(:files) { ["Q4_K_M/big-Q4_K_M-00001-of-00002.gguf", "Q4_K_M/big-Q4_K_M-00002-of-00002.gguf"] }

    it "refuses it" do
      expect { pull("me/big") }.to raise_error(/split model/)
    end
  end

  %w[../x-Q4_K_M.gguf dir/x-Q4_K_M.gguf C:x-Q4_K_M.gguf -x-Q4_K_M.gguf NUL.Q4_K_M.gguf].each do |unsafe|
    context "with a file named #{unsafe}" do
      let(:files) { [unsafe] }

      it "refuses it" do
        expect { pull("me/tiny") }.to raise_error(/refusing the file name/)
      end
    end
  end

  context "without a checksum in the listing" do
    let(:sha256) { nil }

    it "refuses the model" do
      expect { pull("me/tiny") }.to raise_error(/does not say the size and SHA-256/)
    end
  end

  it "refuses a listing without a full commit or a size" do
    model = Elelem::Llama::PinnedModels::SMALL

    [{ commit: "main" }, { bytes: 0 }, { bytes: "9" }].each do |change|
      expect { model.with(**change).verified }.to raise_error(/does not say/)
    end
    expect(model.verified).to eq(model)
  end

  it "refuses what is not a repository" do
    ["tiny", "../tiny", "me/..", "me/tiny:a/b", "me/tiny:x.gguf", "https://x/y", "me/tiny\n"].each do |name|
      expect { Elelem::Llama::ModelReference.parse(name) }.to raise_error(ArgumentError)
    end
  end

  it "pins the 4B model, or the 2B model below 12 GiB or where the memory is unknown" do
    memory = Struct.new(:total)
    chosen = [12 * (1024**3), (12 * (1024**3)) - 1, nil].map { Elelem::Llama::PinnedModels.for(memory.new(it)).file }

    expect(chosen).to eq(%w[Qwen3.5-4B-Q4_K_M.gguf Qwen3.5-2B-Q4_K_M.gguf Qwen3.5-2B-Q4_K_M.gguf])
  end

  it "links by hard link where symlinks are not allowed, and replaces that link later" do
    put(saved("a.gguf"), "a")
    put(saved("b.gguf"), "b")
    default = Elelem::Llama::DefaultModel.new(dir: @dir, symlink: ->(*) { raise Errno::EACCES })

    default.point_at(saved("a.gguf"))
    default.point_at(saved("b.gguf"))

    expect(File.read(saved("default.gguf"))).to eq("b")
    expect(File.read(saved("a.gguf"))).to eq("a")
  end
end
