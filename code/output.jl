function printSuccess(message)
    println(string("✔ ", message))
end

function printError(message)
    println(stderr, string("❗ ", message))
end
